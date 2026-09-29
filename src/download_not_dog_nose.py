import os
import time
import random
import requests


# ============================================================
# PATH
# ============================================================

BASE_DIR = (
    r"E:\canine-nose-print-recognition\ai-model"
    r"\dataset\nose_validation\not_dog_nose"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = {
    "humans": 60,
    "other_animals": 120,
    "random_objects": 120
}


# ============================================================
# SEARCH TERMS
# ============================================================

CATEGORIES = {

    "humans": [
        "person face",
        "people face",
        "human face"
    ],

    "other_animals": [
        "cat face",
        "cat animal",
        "horse face",
        "horse animal",
        "cow face",
        "cow animal",
        "bird face",
        "bird animal",
        "goat face",
        "goat animal",
        "rabbit face",
        "rabbit animal",
        "sheep face",
        "sheep animal"
    ],

    "random_objects": [
        "car front",
        "chair",
        "bottle",
        "laptop",
        "backpack",
        "shoe",
        "camera",
        "phone",
        "helmet",
        "toy"
    ]
}


# ============================================================
# WIKIMEDIA API
# ============================================================

API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent":
        "CanineNosePrintRecognition/1.0 "
        "(educational research project)"
}


# ============================================================
# REQUEST SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    HEADERS
)


# ============================================================
# COUNT EXISTING IMAGES
# ============================================================

def get_existing_images(folder):

    if not os.path.exists(folder):
        return []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )

    files = []

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
            continue

        if not filename.lower().endswith(
            valid_extensions
        ):
            continue

        # Ignore empty/corrupt zero-byte files
        if os.path.getsize(path) == 0:
            continue

        files.append(filename)

    return files


# ============================================================
# SEARCH WIKIMEDIA
# ============================================================

def search_images(
    search_term,
    limit=30
):

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 500,
        "format": "json"
    }

    try:

        response = session.get(
            API_URL,
            params=params,
            timeout=30
        )

        # Wikimedia rate limit
        if response.status_code == 429:

            print(
                "Wikimedia rate limit reached. "
                "Waiting 10 seconds..."
            )

            time.sleep(10)

            return []

        response.raise_for_status()

        data = response.json()

        results = []

        for page in (
            data.get("query", {})
            .get("pages", {})
            .values()
        ):

            imageinfo = page.get(
                "imageinfo"
            )

            if not imageinfo:
                continue

            url = imageinfo[0].get(
                "thumburl"
            )

            if url:
                results.append(url)

        return results

    except Exception as error:

        print(
            f"Search failed for "
            f"'{search_term}': {error}"
        )

        return []


# ============================================================
# FIND UNUSED FILENAME
# ============================================================

def get_next_filename(folder):

    number = 1

    while True:

        filename = (
            f"image_{number:04d}.jpg"
        )

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.exists(path):

            return filename

        number += 1


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(
    url,
    folder
):

    try:

        response = session.get(
            url,
            timeout=30
        )

        if response.status_code == 429:

            print(
                "Download rate limited. "
                "Waiting 10 seconds..."
            )

            time.sleep(10)

            return False

        if response.status_code != 200:

            return False

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if "image" not in content_type:

            return False

        image_data = response.content

        if len(image_data) < 1000:

            return False

        filename = get_next_filename(
            folder
        )

        path = os.path.join(
            folder,
            filename
        )

        # ----------------------------------------------------
        # WRITE FILE
        # ----------------------------------------------------

        with open(
            path,
            "wb"
        ) as file:

            file.write(
                image_data
            )

        # ----------------------------------------------------
        # VERIFY FILE
        # ----------------------------------------------------

        if not os.path.exists(path):

            print(
                "File verification failed."
            )

            return False

        if os.path.getsize(path) < 1000:

            print(
                "Downloaded file is too small."
            )

            os.remove(path)

            return False

        print(
            f"Saved: {filename}"
        )

        return True

    except Exception as error:

        print(
            "Download error:",
            error
        )

        return False


# ============================================================
# COLLECT CATEGORY
# ============================================================

def collect_images(
    folder_name,
    search_terms,
    target
):

    folder = os.path.join(
        BASE_DIR,
        folder_name
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    existing_files = get_existing_images(
        folder
    )

    existing_count = len(
        existing_files
    )

    print()
    print("=" * 60)
    print(
        f"Category: {folder_name}"
    )
    print(
        f"Existing: {existing_count}"
    )
    print(
        f"Target:   {target}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # ALREADY COMPLETE
    # --------------------------------------------------------

    if existing_count >= target:

        print(
            f"{folder_name} already has "
            f"{existing_count} valid images."
        )

        return existing_count

    needed = (
        target - existing_count
    )

    print(
        f"Need {needed} more images."
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    all_urls = []

    for term in search_terms:

        print(
            f"Searching: {term}"
        )

        urls = search_images(
            term,
            limit=30
        )

        all_urls.extend(
            urls
        )

        # Small delay between searches
        time.sleep(2)

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    all_urls = list(
        set(all_urls)
    )

    random.shuffle(
        all_urls
    )

    print(
        f"Unique URLs found: "
        f"{len(all_urls)}"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    downloaded = 0

    for url in all_urls:

        if downloaded >= needed:

            break

        success = download_image(
            url,
            folder
        )

        if success:

            downloaded += 1

            print(
                f"Progress: "
                f"{downloaded}/{needed}"
            )

        # Delay between downloads
        time.sleep(1)

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    final_files = get_existing_images(
        folder
    )

    final_count = len(
        final_files
    )

    print()
    print(
        f"{folder_name} FINAL COUNT: "
        f"{final_count}/{target}"
    )

    return final_count


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 60)
print("NEGATIVE DATASET DOWNLOADER")
print("=" * 60)

total = 0

for folder_name, target in TARGETS.items():

    count = collect_images(
        folder_name,
        CATEGORIES[folder_name],
        target
    )

    total += count


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINAL DATASET COUNT")
print("=" * 60)

for folder_name, target in TARGETS.items():

    folder = os.path.join(
        BASE_DIR,
        folder_name
    )

    count = len(
        get_existing_images(folder)
    )

    print(
        f"{folder_name:20s}: "
        f"{count}/{target}"
    )

print("-" * 60)

print(
    f"TOTAL: {total}/300"
)

print("=" * 60)