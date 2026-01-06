from datetime import datetime
def create_system_date_and_shop_prompt():
    date_now = datetime.now().strftime("%d/%m/%Y")
    text = f"""
You extract date and shop name from receipt text.
Your task is to find date and shop name.

Return ONLY valid JSON matching the provided schema. No tables, no markdown, no extra text.
Return JSON EXACTLY AS:
```json
{{
    "date_str": string or null,
    "shop_name": string or null
}}
```
The date MUST BE EXACTLY IN FORMAT %d/%m/%Y (day first, 24h time, leading zeros)

GOLDEN RULES (STRICT):
- If multiple dates appear, select the most recent one (today is {date_now})
- If no date is found, set "date_str" to today's date.
- If no shop name is found or you are not certain, set "shop_name" to null.
- DO NOT invent values, if you are not certain, ignore field.
- RETURN ONLY REQUESTED JSON - Do not return a JSON array at the top level.
"""
    return text