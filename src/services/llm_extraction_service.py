from src.prompts.system_extract_prompt import create_extract_system_prompt
from src.prompts.system_categorize_prompt import create_categorize_prompt
from src.prompts.system_date_and_shop_extract_prompt import create_system_date_and_shop_prompt
from dataclasses import dataclass
from src.utils.config_loader import load_settings, load_categories
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from src.utils.dict_handler import merge_dicts
from datetime import datetime
import cv2
import numpy as np



@dataclass(frozen=True)
class _Prompts:
    pr_categorize: str = create_categorize_prompt()
    pr_extract: str = create_extract_system_prompt()
    pr_datename: str = create_system_date_and_shop_prompt()

class _ExtractProduct(BaseModel):
    name: str
    quantity: Optional[float] = None
    price_unit: Optional[float] = None
    promotion: Optional[float] = None
    price_all: Optional[float] = None

class _ExtractReceipt(BaseModel):
    products: List[_ExtractProduct]

class _MergedProduct(BaseModel):
    name: str
    quantity: Optional[float] = None
    price_unit: Optional[float] = None
    promotion: Optional[float] = None
    price_all: Optional[float] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None

class _MergedReceipt(BaseModel):
    products: List[_MergedProduct]

class _ReceiptMetadata(BaseModel):
    date_str: Optional[str] = None
    shop_name: Optional[str] = None

class LLMExtractionService:
    def __init__(self):
        settings = load_settings()
        pr = _Prompts()
        self.system_prompt_categorize = pr.pr_categorize
        self.system_prompt_extract = pr.pr_extract
        self.system_prompt_datename = pr.pr_datename

        base_url = settings['client_url']
        api_key = settings['client_api_key']
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        self.model = settings['model_name']

        extract_settings = settings['model_settings_extract']
        categorize_settings = settings['model_settings_categorize']

        self.extract_settings = merge_dicts(extract_settings)
        self.categorize_settings = merge_dicts(categorize_settings)

        self.categories = load_categories()

        categories_list = self.categories['categories']
        self.categories_set = (set(categories_list.keys()))
        self.subcategories_set = {item for sublist in categories_list.values() for item in sublist}

    def _categorize_products_preprocess(self, products):
        product_list = ""
        for product in products.products:
            product_list += str(product) + '\n'
        return product_list


    def get_products(self, product_string, return_as_dict=False):
        try:
            resp = self.client.responses.parse(
                model = self.model,
                input = [
                    {"role": "system", "content": str(self.system_prompt_extract)},
                    {"role": "user", "content": "REMEMBER - USE VALID SCHEMA.\n\n" + str(product_string)}
                ],
                text_format= _ExtractReceipt ,
                **self.extract_settings,
            )
        except:
            print("Extract products - Didn't get a valid response from LLM")
            return None
        output_resp = resp.output_parsed
        if return_as_dict:
            return output_resp.model_dump()
        return output_resp

    def get_categories(self, products: _ExtractReceipt, return_as_dict=False):
        product_string = self._categorize_products_preprocess(products)
        try:
            resp = self.client.responses.parse(
                model = self.model,
                input = [
                    {"role": "system", "content": str(self.system_prompt_categorize)},
                    {"role": "user", "content": "REMEMBER - USE VALID SCHEMA.\n\n" + product_string}
                ],
                text_format = _MergedReceipt,
                **self.categorize_settings,
            )
        except:
            print("Extract categories - Didn't get a valid response from LLM")
            return None
        categorized_receipt = resp.output_parsed

        # postprocess
        for product in categorized_receipt.products:
            if product.category not in self.categories_set:
                print(f'wrong category for {product.name}')
                product.category = 'N/A'
                product.subcategory = 'N/A'

            elif product.subcategory not in self.subcategories_set:
                print(f'wrong subcategory for {product.name}')
                product.subcategory = 'N/A'
                product.category = 'N/A'

        if return_as_dict:
            return categorized_receipt.model_dump()

        return categorized_receipt

    def extract_date_shop_name(self, ocr_scan):
        try:
            resp = self.client.responses.parse(
                model = self.model,
                input = [
                    {"role": "system", "content": str(self.system_prompt_datename)},
                    {"role": "user", "content": "REMEMBER - USE VALID SCHEMA.\n\n" + str(ocr_scan)},
                ],
                text_format = _ReceiptMetadata,
                **self.categorize_settings,
            )
        except:
            print("Extract date shop name - Didn't get a valid response from LLM")
            return None
        out_resp = resp.output_parsed

        out_name = out_resp.shop_name
        if type(out_name) == str:
            out_name = out_name.lower()
        else:
            out_name = 'N/A'

        unix_time = (int(datetime.strptime(out_resp.date_str, '%d/%m/%Y').timestamp()))

        out_dict = {
            'shop_name': out_name,
            'unix_time': unix_time,
        }
        return out_dict


    def extract_and_categorize(self, ocr_output:str, test=False, return_as_dict=False):
        if test:
            import pickle
            with open('products.txt', 'rb') as f:
                categorized_receipt = pickle.load(f)
        else:
            products = self.get_products(ocr_output)
            categorized_receipt = self.get_categories(products)



        print(categorized_receipt)
        if return_as_dict:
            return categorized_receipt.model_dump()

        return categorized_receipt


if __name__ == '__main__':
    import pickle
    def save_products(products, fname='products.txt'):
        with open(fname, 'wb') as f:
            pickle.dump(products, f)

    test_str = """
 17 rue des Glairons
 FR-38400ST MARTIN D'HERES
 Ticket de vente
 Article
 P.U.EUR Qté
 EUR
 Crème entière liquid
 2,65 2
 5,30 A T
 Rem Crème
 -1,32
 Emincé de porc
 3,99 1
 3,99 A
 Rem Emincé de porc
 -0,80
 Banane vrac
 1,14 A T
 0,880 kg X 1,29
 EUR/kg
 Réduction Lidl Plus
 -0,39
 Gouda jeune en tranc
 3,35 2
 6,70 A T
 Mélange gourmand
 1,45 2
 2,90 A T
 Lait de COCO
 2,07 1
 2,07 A T
 St Marc 82+42
 2,39 1
 2,39 B
 Jambon de Paris
 2,39 1
 2,39 A T
 Orange à dessert 2kg
 3,59 1
 3,59 A T
 Vinaigre d'alcool bl
 0,42 1
 0,42 A T
 Nettoyant WC pin
 1,16 1
 1,16 B
 Baguette 250g
 0,29 1
 0,29 A T
 Chipolatas longue
 4,79 1
 4,79 A T
 Mir Vaisselle Framb.
 1,36 1
 1,36 B
 Mini poivron mix dou
 0,99 1
 0,99 A T
 Ail 250 g
 1,99 1
 1,99 A T
 Pamplemousse 3 fruit
 2,29 1
 2,29 A T
 Beurre moulé
 4,79 1
 4,79 A T
 Smoothie familial
 2,72 1
 2,72 A T
 Herta Knacki Poulet
 2,09 1
 2,09 A T
 Pinot Blanc
 3,99 1
 3,99 B
 Figue fraiche
 0,69 3
 2,07 A T
 Basilic Bio le pot
 1,79 1
 1,79 A T
 Echalote 500g
 1,19 1
 1,19 A T
 Empanadas poulet C.
 1,29 2
 2,58 E T
 Focaccia brie
 1,39 1
 1,39 E T
 Nombre de lignes: 26
 A payer
 63,86
 59,36 HT
 Total éligible TR (T) :
 51,77
 Carte
 63,86
 Total Promotion
 2,51
 TVA Taux
 MONT.TTC MONT.TVA
 TOTAL HT
 A
 5,5%
 50,99
 2,66
 48,33
 B
 20%
 8,90
 1,48
 7,42
 E
 10%
 3,97
 0,36
 3,61
 Avec Lidl Plus,
 vous avez économisé 0,39 EUR
"""
    """
    service = LLMExtractionService()
    with open('metadata.txt', 'rb') as f:
        metadata = pickle.load(f)
    with open('products.txt', 'rb') as f:
        products = pickle.load(f)
    from dataclasses import asdict

    out_dict = dict()
    inside_dict = dict()
    receipt_dict = products.model_dump()
    print(receipt_dict)
    for product in receipt_dict['products']:
        print(product)
    """

    service = LLMExtractionService()
    out = service.extract_and_categorize(test_str, return_as_dict=True)
    out_metadata = service.extract_date_shop_name(test_str)
    save_products(out, 'products.txt')
    save_products(out_metadata, 'metadata.txt')
    print(out)