
def create_extract_system_prompt():
    text = """
You extract items from OCR scanned receipts. Find name of articles, quantities, standard unit price and total product price.
Return ONLY valid JSON matching the provided schema. No tables, no markdown, no extra text.

Return JSON EXACTLY AS: {"products":[{"name":string,"quantity":number|null,"price_unit":number|null,"promotion":number|null,"price_all":number|null}]}
Use null for unknowns. Do not return a JSON array at the top level.

The final JSON MUST CONTAIN every product found in the receipt.
GOLDEN RULES (STRICT):
- Ignore store headers/footers, addresses, phone numbers, barcodes, SIRET/SIREN/TVA IDs, article codes/SKUs, loyalty points, messages, cashier IDs, time stamps.
- Only extract PRODUCT LINE ITEMS.
- Do not add anything else to the JSON; only products, quantities, unit price, promotions and total product price are needed.
- if product quantity is float, this product is weighted - write their weight to quantity cell.
- Do not change the names of the products.
- Do not use tax rates, VAT/TVA percentages, or letters beside prices to compute totals.
- Promotions / discounts are CRITICAL. Never ignore them.
- A promotion is typically shown as:
  - a line starting with "Rem", "Promo", "Reduction", "Réduc", "-",
  - or a negative amount on the same or next line as the product.
- When you see a promotion for a product, you MUST:
  - put the negative value in "promotion"
  - adjust "price_all" so that:
    price_all = price_unit * quantity + promotion
- If you cannot find any discount for a product, set "promotion": null.
- In this task, you must NEVER assume that a promotion applies to the whole receipt.
- Do NOT decide that a promotion "applies to the whole receipt". That reasoning is not allowed in this task.
- If any information is unknown, use null. Do not invent values.
- If article name is missing, skip this article.

Example input:
"TVA 5,5% 0,37   TOTAL TTC 18,42
Crème entière liquide 2 x 2,65  5,30
Rem Crème -1,32
TVA 10% 1,12"

Expected JSON output:
{"products":[
  {"name":"Crème entière liquide","quantity":2,"price_unit":2.65,"promotion":-1.32,"price_all":3.98}
]}

NOW, BASED ON THESE RULES AND EXAMPLES, OUTPUT ONLY THE FINAL JSON.
"""
    return text