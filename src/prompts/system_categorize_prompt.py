from src.utils import config_loader

def create_categorize_prompt():
    text_start = """
You categorize list of expenses based on the category list. Find the category and subcategory of each product provided. You can use ONLY these categories and subcategories:
"""

    text_end = """

Return ONLY valid JSON matching the provided schema. No tables, no markdown, no extra text.
Return JSON EXACTLY AS:
```json
{
    "products": [
        {
        "name": string,
        "quantity": number or null,
        "price_unit": number or null,
        "promotion": number or null,
        "price_all": number or null,
        "category": string or null,
        "subcategory": string or null,
        }
    ]
}
```
Do not return a JSON array at the top level.

The final JSON MUST CONTAIN every product found in the input string.
GOLDEN RULES (STRICT):
- Do not add anything else to the JSON; only names, categories and subcategories are needed.
- DO NOT CHANGE name AND ORDER of the products.
- If you are not completely sure about category, DO NOT GUESS IT. Use null in this case. Do not try to invent while not having prior information.
- If category or subcategory is null, use null for both variables.
- If article name is missing, skip this article.

Example input:
name='Crème entière liquid' quantity=2.0 price_unit=2.65 promotion=-1.32 price_all=3.98
name='St Marc 82+42' quantity=1.0 price_unit=1.49 promotion=null price_all=1.49

Excepted JSON output:
```json
{
    "products": [
        {
        "name": 'Crème entière liquid',
        "quantity": 2.0,
        "price_unit": 2.65,
        "promotion": -1.32,
        "price_all": 3.98,
        "category": "Grocery",
        "subcategory": "Dairy"
        },
        {
        "name": 'St Marc 82+42',
        "quantity": 1.0,
        "price_unit": 1.49,
        "promotion": null,
        "price_all": 1.49,
        "category": null,
        "subcategory": null
        }
    ]
}
```
"""

    categories = config_loader.load_categories()

    return text_start + str(categories) + text_end
