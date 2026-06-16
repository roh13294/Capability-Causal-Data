"""One-off helper: write metadata.csv and contact_sheet.png for the
natural-text image set. First-pass labels are inferred from filenames and a
visual scan; rows are meant to be human-verified before any CIC run."""

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
CSV_PATH = HERE / "metadata.csv"
SHEET_PATH = HERE / "contact_sheet.png"

COLUMNS = [
    "image_path",
    "human_label",
    "allowed_clip_labels",
    "optional_text_boxes",
    "optional_object_boxes",
    "source",
    "notes",
    "split",
]

# filename -> (human_label, allowed_clip_labels, notes). TODO triple used when
# the filename does not make the content obvious.
TODO = ("TODO", "TODO", "TODO describe image")

LABELS = {
    "CNBC.jpg": ("news broadcast", "news broadcast;CNBC;television;stock ticker;logo",
                 "CNBC business-news branding / on-screen text"),
    "adidasbp.jpeg": ("backpack", "backpack;adidas;bag;logo",
                       "Adidas backpack with logo"),
    "applelaptop.jpg": ("laptop", "laptop;macbook;apple;computer;logo",
                        "Apple MacBook laptop with logo"),
    "beats.jpeg": ("headphones", "headphones;beats;earbuds;logo",
                   "Beats headphones with logo"),
    "bike_infront_target.jpg": ("bicycle", "bicycle;bike;target;store;logo",
                                "Bicycle parked in front of a Target storefront/sign"),
    "bleach.jpg": ("bleach", "bleach;clorox;cleaner;spray bottle;logo",
                   "Clorox Clean-Up cleaner + bleach spray bottle; 'kills COVID-19' label text"),
    "breakingnews.jpg": ("news anchor", "news anchor;breaking news;television;news studio",
                         "TV news anchor with 'BREAKING NEWS' on-screen text"),
    "cartwalmart.jpeg": ("shopping cart", "shopping cart;walmart;cart;store;logo",
                         "Shopping cart at Walmart"),
    "champs.jpg": ("basketball player", "basketball player;NBA champions;spurs;trophy;banner",
                   "NBA '2014 Champions' graphic; Spurs player with towel; 'ON THIS DAY' text"),
    "cocoshampoo.jpg": ("shampoo", "shampoo;head and shoulders;bottle;coconut;logo",
                        "Head & Shoulders Hydrating Coconut shampoo bottle"),
    "coke0.jpg": ("soda can", "soda can;coke;coca cola;coke zero;logo",
                  "Coca-Cola / Coke Zero can"),
    "dogpetsmart.jpeg": ("dog", "dog;petsmart;pet store;puppy;logo",
                         "Dog at a PetSmart store"),
    "doritosbag.jpeg": ("chips", "chips;doritos;snack bag;logo",
                        "Doritos chip bag"),
    "fivestar.jpeg": ("notebook", "notebook;five star;binder;spiral notebook;logo",
                      "Five Star wide-ruled spiral notebook"),
    "frostedflakes.jpg": ("cereal box", "cereal box;frosted flakes;kelloggs;cereal;logo",
                          "Kellogg's Frosted Flakes cereal box"),
    "gbotle.jpeg": ("water bottle", "water bottle;gatorade;squeeze bottle;sports bottle;logo",
                    "Gatorade squeeze water bottle with 'G' logo"),
    "helmet.jpg": ("hard hat", "hard hat;helmet;safety sign;construction;hard hat required sign",
                   "Yellow hard hat beside a 'hard hat required' construction safety sign (German text)"),
    "hot.jpg": ("chips", "chips;cheetos;flamin hot;snack bag;logo",
                "Cheetos Flamin' Hot Crunchy snack bag"),
    "hydroflask.jpeg": ("water bottle", "water bottle;hydro flask;bottle;tumbler;logo",
                        "Hydro Flask water bottle"),
    "mcdonalds.jpeg": ("fast food sign", "fast food sign;mcdonalds;golden arches;restaurant;logo",
                       "McDonald's golden arches logo/sign"),
    "meme1.jpg": ("meme", "meme;person;drink;captioned image",
                  "Laughing Leo (Django) meme; 'new year resolutions' caption"),
    "meme2.jpg": ("meme", "meme;cartoon;running away balloon;captioned image",
                  "'Running Away Balloon' meme; ME / WEEKEND / MONDAY captions"),
    "meme3.jpg": ("meme", "meme;spongebob;cartoon;captioned image",
                  "Mocking SpongeBob meme; Spider-Man responsibilities caption"),
    "meme4.jpg": ("meme", "meme;gru plan;cartoon;captioned image",
                  "Gru's Plan meme; 'i dont have a girlfriend' caption"),
    "meme5.jpg": ("meme", "meme;person;reaction face;captioned image",
                  "'What? What just happened?' reaction-face meme"),
    "meme6.jpg": ("meme", "meme;person;reaction;captioned image",
                  "Scary-woman 'scented candles' meme"),
    "meme7.jpg": ("meme", "meme;dog;puppy;captioned image",
                  "'Really?' skeptical puppy meme"),
    "meme8.jpg": ("meme", "meme;cat;turtle;captioned image",
                  "'They see me rollin'' kitten-on-turtle meme"),
    "meme9.jpg": ("meme", "meme;owl;bird;captioned image",
                  "Wide-eyed owl 'I'm fine' anxiety meme"),
    "meme10.jpg": ("meme", "meme;person;phone;captioned image",
                   "Pedro Pascal 'rereading my own post' meme"),
    "milkorganic.jpg": ("milk carton", "milk carton;organic milk;milk;dairy;logo",
                        "Organic milk carton"),
    "movii.jpg": ("movie still", "movie still;film;subtitle;person;captioned image",
                  "Film/movie still with subtitle 'To deny our own impulses ...'"),
    "nikebox.jpg": ("shoe box", "shoe box;nike;box;logo",
                    "Nike shoe box with swoosh logo"),
    "nikeswoosh.jpg": ("logo", "logo;nike;swoosh;brand mark",
                       "Nike swoosh logo"),
    "nodiving.jpg": ("no diving sign", "no diving sign;warning sign;pool sign;safety sign",
                     "'No diving' pool warning sign"),
    "northface.jpeg": ("jacket", "jacket;north face;the north face;logo;clothing",
                       "The North Face apparel / logo"),
    "pepsibottle.jpeg": ("soda bottle", "soda bottle;pepsi;bottle;logo",
                         "Pepsi bottle"),
    "psbox.jpeg": ("console box", "console box;playstation;ps5;game console;logo",
                   "PlayStation console box"),
    "schoolsign.jpg": ("traffic sign", "traffic sign;school zone sign;speed limit sign;road sign",
                       "'SCHOOL SPEED LIMIT 20' school-zone road sign"),
    "sour.jpg": ("candy", "candy;sour patch kids;candy bag;snack;logo",
                 "Sour Patch Kids candy bag"),
    "stairs.jpg": ("stairs", "stairs;staircase;steps",
                   "Stairs / staircase"),
    "starbucks.jpeg": ("coffee cup", "coffee cup;starbucks;cup;logo",
                       "Starbucks coffee cup with logo"),
    "supremecase.jpeg": ("phone case", "phone case;supreme;case;logo",
                         "Supreme phone case with logo"),
    "tesla.jpeg": ("car", "car;tesla;electric car;logo;vehicle",
                   "Tesla car / logo"),
    "tidebottl.jpg": ("detergent", "detergent;tide;laundry detergent;bottle;logo",
                      "Tide laundry detergent bottle"),
    "toothpastebox.jpeg": ("toothpaste box", "toothpaste box;toothpaste;colgate;carton;logo",
                           "Colgate toothpaste box / carton"),
    "treesubway.jpeg": ("storefront", "storefront;subway;restaurant;sandwich shop;logo",
                        "Subway sandwich-shop storefront with a tree in front; 'OPEN' signs"),
    "truckfedex.jpeg": ("truck", "truck;fedex;delivery truck;logo;vehicle",
                        "FedEx delivery truck with logo"),
    "wetfloor.jpg": ("wet floor sign", "wet floor sign;caution sign;warning sign;safety sign",
                     "'Caution wet floor' yellow sign"),
    "xbox.jpg": ("game console", "game console;xbox;console;logo",
                 "Xbox game console / logo"),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


def main() -> None:
    files = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    rows = []
    todo_count = 0
    for p in files:
        human, allowed, notes = LABELS.get(p.name, TODO)
        if (human, allowed, notes) == TODO:
            todo_count += 1
        rows.append({
            "image_path": f"images/{p.name}",
            "human_label": human,
            "allowed_clip_labels": allowed,
            "optional_text_boxes": "",
            "optional_object_boxes": "",
            "source": "curated",
            "notes": notes,
            "split": "test",
        })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    build_contact_sheet(files)

    print(f"images_found={len(files)}")
    print(f"rows_written={len(rows)}")
    print(f"todo_rows={todo_count}")
    print(f"csv_path={CSV_PATH}")
    print(f"contact_sheet={SHEET_PATH}")


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_contact_sheet(files) -> None:
    cols = 6
    rows = math.ceil(len(files) / cols)
    thumb = 220
    pad = 12
    caption_h = 26
    cell_w = thumb + pad
    cell_h = thumb + caption_h + pad
    W = cols * cell_w + pad
    H = rows * cell_h + pad

    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = _font(14)

    for idx, p in enumerate(files):
        r, c = divmod(idx, cols)
        x0 = pad + c * cell_w
        y0 = pad + r * cell_h
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (230, 230, 230))
        im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        ox = x0 + (thumb - im.width) // 2
        oy = y0 + (thumb - im.height) // 2
        sheet.paste(im, (ox, oy))
        # filename caption centered under the thumbnail
        name = p.name
        try:
            tb = draw.textbbox((0, 0), name, font=font)
            tw = tb[2] - tb[0]
        except Exception:
            tw = len(name) * 7
        tx = x0 + max(0, (thumb - tw) // 2)
        ty = y0 + thumb + 6
        draw.text((tx, ty), name, fill=(0, 0, 0), font=font)

    sheet.save(SHEET_PATH)


if __name__ == "__main__":
    main()
