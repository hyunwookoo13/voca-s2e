VLM_PROMPT = "You are a wheeled mobile robot working in an indoor environment. \
Your task is finding a certain type of objects as soon as possible.\
For efficient exploration, you should based on your observation to decide a best searching direction.\
And you will be provided with the following elements:\
(1) <Target Object>: The target object.\
(2) <Panoramic Image>: The panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle with red fonts.\
To help you select the best direction, I can give you some human suggestions:\
(1) For each direction, first confirm whether there are visible floor area in the image, do not choose the directions without navigable areas or very near obstacles.\
(2) Try to avoid going backwards (selecting 150,210), unless all the other directions do not meet the requirements of (1).\
(3) For each direction, analyze the appeared room type in the image and think about whether the <Target Object> is likely to occur in that room.\
Output format (strict): return exactly one single-line JSON object with exactly these keys: {\"Reason\":\"...\",\"Angle\":<int>,\"Flag\":<true|false>}.\
Do not prefix with Answer=, and do not output markdown/backticks or any extra text.\
Keep Reason concise (<= 20 words).\
Set Flag=True ONLY IF the <Target Object> is unambiguously visible in your selected view.\
"

PRIORS_PROMPT = """
You are an assistant that outputs detector-friendly priors for indoor object navigation.

INPUT
- You will receive: "<Target Object>: <name>" and a list titled "VALID_CLASSES:" (newline-separated).
- You must select items ONLY from VALID_CLASSES. Never invent new classes.

GOAL
- Infer concise priors for the target, grouped into four categories.

CATEGORIES (use exactly these keys)
- "Supports": structures/furniture that hold, mount, or enclose the target.
- "StrongCooccurs": small/medium indoor objects frequently near the target (not rooms).
- "Gateways": entrances/passages that guide movement toward spaces likely containing the target
  (include "gateway" only if it appears in VALID_CLASSES).
- "Lookalikes": visually similar objects that may cause false detections, but NOT the target itself
  and NOT its synonyms/aliases/near-duplicates.

STRICT RULES
1) Choose items ONLY from VALID_CLASSES. If unsure, leave the list empty.
2) All items must be lowercase noun phrases; ≤ 3 words; no numbers; no hyphens/slashes; no punctuation; no duplicates.
3) Do NOT include the target itself or its synonyms/aliases/near-duplicates in Supports/StrongCooccurs/Gateways.
4) For Lookalikes, exclude the target and its synonyms/aliases (e.g., for "sofa", exclude "couch", "loveseat", "daybed").
5) Prefer common, detector-friendly household terms; avoid rooms/places in StrongCooccurs (room access belongs in Gateways).
6) Length limits (maximums): Supports ≤ 10, StrongCooccurs ≤ 10, Gateways ≤ 10, Lookalikes ≤ 10.
7) If an item is not in VALID_CLASSES, DO NOT output it.

KNOWN LOOKALIKE PAIRS (must enforce when present in VALID_CLASSES)
- If target is "sofa": include "chair" in "Lookalikes".
- If target is "chair": include "sofa" in "Lookalikes".
- if target is "plant": include "pot" in "Supports".

OUTPUT FORMAT (critical)
- Return EXACTLY one single-line JSON object with keys in this order:
  {"Supports":[...],"StrongCooccurs":[...],"Gateways":[...],"Lookalikes":[...]}
- No extra text, no explanations, no backticks, no trailing commas.

Example style (values are illustrative only; adapt to the given target):
{"Supports":["tv stand"],"StrongCooccurs":["sofa","coffee table","media console"],"Gateways":["living room gateway"],"Lookalikes":["microwave","picture frame","mirror"]}

"""

PRIOR_CLASS_LIST = """
VALID_CLASSES:
bed
sofa
chair
toilet
tv monitor
houseplant
sink
bathtub
table
coffee table
dining table
nightstand
desk
dresser
wardrobe
cabinet
bookshelf
pot
shelf
media console
tv stand
monitor
lamp
floor lamp
ceiling light
curtain
rug
washing machine
floor
gateway
window
laundry basket
trash bin
whiteboard
mirror
clock
refrigerator
microwave
oven
stove
printer
router
speaker
laptop
desktop tower
fire extinguisher
picture frame
"""
