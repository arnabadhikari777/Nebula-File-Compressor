"""
Nebula File Compressor
=======================
An all-in-one, fully in-memory file transformation suite built with Flask.

Modules:
    1. Smart Photo & Signature Resizer  -> dimension presets + exact target-KB matching
    2. Universal Format Converter       -> image<->image, multi-image -> PDF, PDF -> images
    3. PDF Size Optimizer & Compressor  -> strips metadata & compresses content streams

Design principle: STRICTLY NO local disk writes. Every uploaded file is read into
memory and every generated file is streamed back to the client via io.BytesIO.
This keeps the app stateless and safe to run on ephemeral/read-only cloud
filesystems such as Render's free tier.

Website  : Nebula File Compressor
Created by: Arnab Adhikari
"""

import io
import os
import zipfile

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    flash,
    redirect,
    url_for,
)
from PIL import Image
from pypdf import PdfReader, PdfWriter
import fitz  # PyMuPDF - used for fast, dependency-free PDF <-> image rendering

# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)

# The secret key is required for flash messaging. In production on Render,
# set the SECRET_KEY environment variable; a fallback is provided so the app
# still runs out-of-the-box in local/dev environments.
app.secret_key = os.environ.get("SECRET_KEY", "nebula-file-compressor-dev-secret-key")

# Cap total request size to protect the free-tier instance's RAM, since all
# processing happens in memory rather than on disk.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}

# Preset dimensions (width, height) in pixels for common exam / job-form photos.
# Passport size 3.5cm x 4.5cm @ 300 DPI -> (3.5/2.54*300, 4.5/2.54*300) ≈ (413, 531)
PHOTO_PRESETS = {
    "ssc_upsc_photo": (200, 230),
    "signature": (140, 60),
    "passport": (413, 531),
    "custom": None,
}


# ---------------------------------------------------------------------------
# Small, reusable helper functions
# ---------------------------------------------------------------------------
def allowed_file(filename, allowed_extensions):
    """Return True if filename has one of the allowed extensions."""
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def convert_to_rgb_with_white_bg(img):
    """
    Safely convert any image mode (including RGBA / LA / P-with-transparency)
    to a clean flat RGB image with a white background. This is required
    before saving as JPEG or BMP, which do not support an alpha channel.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Use the image's own alpha channel as the paste mask
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


def resize_image_dimensions(img, width, height, maintain_aspect):
    """
    Resize a PIL Image to the given width/height.
    - If maintain_aspect is True, the image is scaled down to FIT inside the
      given box (never upscaled distorted), preserving its original ratio.
    - If False, the image is force-resized to the exact width x height,
      which may stretch/squash the image (useful for strict form templates).
    """
    if maintain_aspect:
        working = img.copy()
        working.thumbnail((width, height), Image.LANCZOS)
        return working
    return img.resize((width, height), Image.LANCZOS)


def image_to_bytes(img, fmt, quality=95):
    """
    Save a PIL Image into an in-memory BytesIO buffer in the requested format.
    Returns the buffer positioned at offset 0, ready to be read or sent.
    """
    buffer = io.BytesIO()
    fmt_normalized = "JPEG" if fmt.upper() in ("JPG", "JPEG") else fmt.upper()

    save_kwargs = {}
    if fmt_normalized == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif fmt_normalized == "PNG":
        save_kwargs["optimize"] = True
    elif fmt_normalized == "WEBP":
        save_kwargs["quality"] = quality

    img.save(buffer, format=fmt_normalized, **save_kwargs)
    buffer.seek(0)
    return buffer


def fit_to_target_size(img, target_kb):
    """
    Iteratively compress a JPEG image so its final byte size is at or below
    target_kb, without needlessly degrading resolution.

    Strategy:
      1. Try progressively lower JPEG quality settings from 95 down to 10
         (in steps of 5). As soon as a quality level produces a file at or
         under the target size, that result is returned immediately -- this
         preserves maximum resolution and quality for the given size budget.
      2. If even quality=10 is still above the target size (common for very
         small targets like 20 KB on a large photo), the image dimensions
         are scaled down by 10% and the quality sweep is repeated. This
         repeats up to 10 times, which is more than enough for typical
         exam-form use cases (e.g. shrinking a 20 KB signature).
      3. If the target still cannot be reached, the smallest buffer produced
         across all attempts is returned so the user always gets a usable
         result rather than an error.

    Returns: (BytesIO buffer, quality_used, final_size_in_bytes)
    """
    target_bytes = target_kb * 1024
    working_img = img.copy()

    # JPEG has no alpha channel -- flatten transparency onto white first.
    if working_img.mode in ("RGBA", "LA", "P"):
        working_img = convert_to_rgb_with_white_bg(working_img)

    smallest_buffer = None
    smallest_size = None
    smallest_quality = 10

    for _ in range(11):  # initial attempt + up to 10 downscale retries
        found_buffer = None
        found_size = None
        found_quality = None

        for quality in range(95, 9, -5):
            buffer = image_to_bytes(working_img, "JPEG", quality=quality)
            size = buffer.getbuffer().nbytes

            # Track the globally smallest result as a safety-net fallback.
            if smallest_size is None or size < smallest_size:
                smallest_buffer, smallest_size, smallest_quality = buffer, size, quality

            if size <= target_bytes:
                found_buffer, found_size, found_quality = buffer, size, quality
                break

        if found_buffer is not None:
            return found_buffer, found_quality, found_size

        # Quality sweep alone wasn't enough -- shrink dimensions and retry.
        w, h = working_img.size
        new_w, new_h = max(1, int(w * 0.9)), max(1, int(h * 0.9))
        if (new_w, new_h) == (w, h) or new_w < 20 or new_h < 20:
            break
        working_img = working_img.resize((new_w, new_h), Image.LANCZOS)

    smallest_buffer.seek(0)
    return smallest_buffer, smallest_quality, smallest_size


def compress_pdf_bytes(reader):
    """
    Build a size-optimized copy of a PDF using pypdf:
      - Recompresses each page's content stream (redundant operators removed).
      - Strips document metadata (author, producer, timestamps, etc.).
      - Deduplicates identical indirect objects (e.g. repeated embedded
        resources) and drops orphaned objects, when supported by the
        installed pypdf version.

    Returns an in-memory BytesIO buffer containing the compressed PDF.
    """
    writer = PdfWriter()

    for page in reader.pages:
        try:
            page.compress_content_streams()
        except Exception:
            # If a page's content stream can't be recompressed for any
            # reason, keep the original page rather than failing the job.
            pass
        writer.add_page(page)

    # Strip all metadata to shave off extra bytes and remove authoring info.
    writer.add_metadata({})

    # Newer pypdf versions expose object-level deduplication; guard for
    # compatibility with older installed versions.
    if hasattr(writer, "compress_identical_objects"):
        try:
            writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
        except Exception:
            pass

    output_buffer = io.BytesIO()
    writer.write(output_buffer)
    output_buffer.seek(0)
    return output_buffer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Render the single-page dashboard. `tab` query param sets active tab."""
    active_tab = request.args.get("tab", "resizer")
    if active_tab not in ("resizer", "converter", "pdf_tools"):
        active_tab = "resizer"
    return render_template("index.html", active_tab=active_tab)


@app.route("/resize", methods=["POST"])
def resize_photo():
    """Module 1: Smart Photo & Signature Resizer with optional target-KB matching."""
    file = request.files.get("photo_file")
    if not file or file.filename == "":
        flash("Please select an image file to resize.", "error")
        return redirect(url_for("index", tab="resizer"))

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        flash("Unsupported image format. Please use JPG, JPEG, PNG, or WEBP.", "error")
        return redirect(url_for("index", tab="resizer"))

    preset = request.form.get("preset", "custom")
    maintain_aspect = request.form.get("maintain_aspect") == "on"
    output_format = request.form.get("output_format", "JPEG").upper()
    target_kb_raw = request.form.get("target_kb", "").strip()

    # Resolve target width/height from preset or custom fields.
    if preset in PHOTO_PRESETS and PHOTO_PRESETS[preset] is not None:
        width, height = PHOTO_PRESETS[preset]
    else:
        try:
            width = int(request.form.get("custom_width", 0))
            height = int(request.form.get("custom_height", 0))
        except ValueError:
            width, height = 0, 0
        if width <= 0 or height <= 0:
            flash("Please provide valid custom width and height in pixels.", "error")
            return redirect(url_for("index", tab="resizer"))

    try:
        img = Image.open(file.stream)
        img.load()
    except Exception:
        flash("The uploaded file could not be read as an image.", "error")
        return redirect(url_for("index", tab="resizer"))

    resized = resize_image_dimensions(img, width, height, maintain_aspect)

    if output_format in ("JPEG", "JPG"):
        resized = convert_to_rgb_with_white_bg(resized)

    if target_kb_raw:
        try:
            target_kb = int(target_kb_raw)
            if target_kb <= 0:
                raise ValueError
        except ValueError:
            flash("Target size (KB) must be a positive whole number.", "error")
            return redirect(url_for("index", tab="resizer"))

        # Exact target-size matching is only meaningful for lossy JPEG output.
        if output_format not in ("JPEG", "JPG"):
            flash("Target file-size matching requires JPEG output — switched automatically.", "info")
            resized = convert_to_rgb_with_white_bg(resized)

        buffer, _quality_used, _final_size = fit_to_target_size(resized, target_kb)
        ext = "jpg"
    else:
        buffer = image_to_bytes(resized, output_format, quality=95)
        ext = "jpg" if output_format in ("JPEG", "JPG") else output_format.lower()

    mimetype = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    return send_file(
        buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"nebula_resized.{ext}",
    )


@app.route("/convert", methods=["POST"])
def convert_format():
    """Module 2a: Convert a single image between JPG / PNG / WEBP / BMP."""
    file = request.files.get("convert_file")
    if not file or file.filename == "":
        flash("Please select an image to convert.", "error")
        return redirect(url_for("index", tab="converter"))

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        flash("Unsupported source image format.", "error")
        return redirect(url_for("index", tab="converter"))

    target_format = request.form.get("target_format", "PNG").upper()
    if target_format not in ("JPEG", "JPG", "PNG", "WEBP", "BMP"):
        flash("Unsupported target format selected.", "error")
        return redirect(url_for("index", tab="converter"))

    try:
        img = Image.open(file.stream)
        img.load()
    except Exception:
        flash("The uploaded file could not be read as an image.", "error")
        return redirect(url_for("index", tab="converter"))

    # JPEG and BMP cannot store transparency -- flatten to white background.
    if target_format in ("JPEG", "JPG", "BMP"):
        img = convert_to_rgb_with_white_bg(img)

    buffer = image_to_bytes(img, target_format, quality=95)
    ext = "jpg" if target_format in ("JPEG", "JPG") else target_format.lower()
    mimetype = "image/jpeg" if ext == "jpg" else f"image/{ext}"
    return send_file(
        buffer,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"nebula_converted.{ext}",
    )


@app.route("/images-to-pdf", methods=["POST"])
def images_to_pdf():
    """Module 2b: Merge multiple uploaded images into a single paginated PDF."""
    files = [f for f in request.files.getlist("pdf_images") if f and f.filename]
    if not files:
        flash("Please select at least one image to build a PDF.", "error")
        return redirect(url_for("index", tab="converter"))

    pil_images = []
    for f in files:
        if not allowed_file(f.filename, ALLOWED_IMAGE_EXTENSIONS):
            flash(f"Skipped unsupported file: {f.filename}", "error")
            continue
        try:
            img = Image.open(f.stream)
            img.load()
            pil_images.append(convert_to_rgb_with_white_bg(img))
        except Exception:
            flash(f"Could not process file: {f.filename}", "error")

    if not pil_images:
        flash("No valid images were found to build a PDF.", "error")
        return redirect(url_for("index", tab="converter"))

    buffer = io.BytesIO()
    first_page, remaining_pages = pil_images[0], pil_images[1:]
    first_page.save(buffer, format="PDF", save_all=True, append_images=remaining_pages)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="nebula_images.pdf",
    )


@app.route("/pdf-to-images", methods=["POST"])
def pdf_to_images():
    """Module 2c: Extract every page of a PDF as PNG/JPG, zipped if multi-page."""
    file = request.files.get("pdf_to_split")
    if not file or file.filename == "":
        flash("Please select a PDF file to extract images from.", "error")
        return redirect(url_for("index", tab="converter"))

    if not allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
        flash("Please upload a valid PDF file.", "error")
        return redirect(url_for("index", tab="converter"))

    output_format = request.form.get("extract_format", "PNG").upper()
    if output_format not in ("PNG", "JPEG", "JPG"):
        output_format = "PNG"
    ext = "jpg" if output_format in ("JPEG", "JPG") else "png"

    try:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        flash("The uploaded file could not be read as a PDF.", "error")
        return redirect(url_for("index", tab="converter"))

    if doc.needs_pass:
        doc.close()
        flash("This PDF is password-protected. Please remove the password before extracting images.", "error")
        return redirect(url_for("index", tab="converter"))

    if doc.page_count == 0:
        doc.close()
        flash("The PDF has no pages to extract.", "error")
        return redirect(url_for("index", tab="converter"))

    # 2x zoom gives crisp, print-quality output without an excessive file size.
    zoom_matrix = fitz.Matrix(2.0, 2.0)
    rendered_pages = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pixmap = page.get_pixmap(matrix=zoom_matrix)
        raw_png_bytes = pixmap.tobytes("png")

        pil_img = Image.open(io.BytesIO(raw_png_bytes))
        if output_format in ("JPEG", "JPG"):
            pil_img = convert_to_rgb_with_white_bg(pil_img)
        page_buffer = image_to_bytes(pil_img, output_format, quality=95)
        rendered_pages.append(page_buffer.getvalue())
    doc.close()

    if len(rendered_pages) == 1:
        single_buffer = io.BytesIO(rendered_pages[0])
        mimetype = "image/jpeg" if ext == "jpg" else "image/png"
        return send_file(
            single_buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"nebula_page.{ext}",
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for index, page_bytes in enumerate(rendered_pages, start=1):
            zip_file.writestr(f"page_{index:03d}.{ext}", page_bytes)
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="nebula_pdf_pages.zip",
    )


@app.route("/compress-pdf", methods=["POST"])
def compress_pdf_route():
    """Module 3: PDF Size Optimizer & Compressor."""
    file = request.files.get("pdf_to_compress")
    if not file or file.filename == "":
        flash("Please select a PDF file to compress.", "error")
        return redirect(url_for("index", tab="pdf_tools"))

    if not allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
        flash("Please upload a valid PDF file.", "error")
        return redirect(url_for("index", tab="pdf_tools"))

    try:
        reader = PdfReader(file.stream)
    except Exception:
        flash("The uploaded file could not be read as a PDF. It may be corrupted or encrypted.", "error")
        return redirect(url_for("index", tab="pdf_tools"))

    if reader.is_encrypted:
        flash("This PDF is password-protected. Please remove the password before compressing.", "error")
        return redirect(url_for("index", tab="pdf_tools"))

    if len(reader.pages) == 0:
        flash("The PDF has no pages to compress.", "error")
        return redirect(url_for("index", tab="pdf_tools"))

    try:
        compressed_buffer = compress_pdf_bytes(reader)
    except Exception:
        flash("An error occurred while compressing this PDF.", "error")
        return redirect(url_for("index", tab="pdf_tools"))

    return send_file(
        compressed_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="nebula_compressed.pdf",
    )


@app.errorhandler(413)
def file_too_large(_error):
    """Friendly error page when the uploaded payload exceeds MAX_CONTENT_LENGTH."""
    flash("The uploaded file(s) exceed the maximum allowed size (25 MB).", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(_error):
    flash("That page does not exist.", "error")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_server_error(_error):
    flash("An unexpected error occurred while processing your file. Please try again.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the PORT environment variable.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
