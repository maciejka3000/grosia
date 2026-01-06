import sqlite3
import pickle
import numpy as np
from src.utils.config_loader import load_settings, load_categories
from src.services.llm_extraction_service import (
_ReceiptMetadata,
_MergedReceipt,
_ExtractReceipt,
_ExtractProduct,
_MergedProduct,
)
import pandas as pd
import pickle
import os
import glob
from src.utils import dict_handler

def to_decimal(value: float | None, dec_places:int = 2) -> str:
    if value is None:
        value = 0
    dec_string = f"{value:.{dec_places}f}"
    return dec_string



class DatabaseService:
    def __init__(self):
        self.settings = load_settings()
        self.categories = load_categories()

        self.db_path = self.settings["db_path"]
        self.image_save_path = self.settings['receipt_save_folder_path']

        # connect to db
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")

        # add N/A to subcategories to uncategorized items
        self.categories['categories']['N/A'] = ['N/A']

        # check/update categories/subcategories
        categories_list = list(self.categories['categories'].keys())


        with self.conn:
            for category in categories_list:

                c_id = self.conn.execute("""
                INSERT INTO category (name) VALUES (?)
                ON CONFLICT(name) DO UPDATE SET name = excluded.name
                RETURNING category_id;
                """, (category, )).fetchone()[0]

                subcategories = self.categories['categories'][category]
                for subcategory in subcategories:
                    self.conn.execute("""
                    INSERT INTO subcategory (category_id, name) VALUES (?, ?)
                    ON CONFLICT(category_id, name) DO NOTHING;
                    """, (c_id, subcategory))




    def __del__(self):
        self.conn.close()


    def add_new_product_line(self, product: dict, receipt_id:int):
        # product: {'name': 'str',
        # 'quantity': float,
        # 'price_unit': float,
        # 'promotion': float | none,
        # 'price_all': float,
        # 'category': str,
        # 'subcategory': str}

        db_item_line_qty = product["quantity"]
        db_item_line_unit_price = product['price_unit']
        db_item_line_promotion = product['promotion']
        db_item_line_total_price = product['price_all']

        db_item_name = product['name']
        db_item_category = product['category']
        db_item_subcategory = product['subcategory']

        # get category and subcategory
        with self.conn:
            category_id = self.conn.execute("""
            SELECT category_id from category where (name) = (?)
            """, (db_item_category, )).fetchone()[0]

            subcategory_id = self.conn.execute("""
            SELECT subcategory_id from subcategory where (name, category_id) = (?, ?)
            """, (db_item_subcategory, category_id)).fetchone()[0]

            # append items
            item_id = self.conn.execute("""
            INSERT INTO item (category_id, subcategory_id, name)
            VALUES (?, ?, ?) ON CONFLICT(name) DO
            UPDATE SET name = excluded.name
            RETURNING item_id;
            """, (category_id, subcategory_id, db_item_name)).fetchone()[0]

            # append itemLines

            self.conn.execute("""
            INSERT INTO item_line (receipt_id, item_id, qty, unit_price, promotion, total_price)
            VALUES (?, ?, ?, ?, ?, ?)    
            """, (receipt_id, item_id, db_item_line_qty, db_item_line_unit_price, db_item_line_promotion, db_item_line_total_price))

        pass

    def add_new_receipt(self, receipt: dict, receipt_metadata: dict, note:str = 'N/A') -> None:

        all_products = receipt["products"]

        db_receipt_shop = receipt_metadata['shop_name']
        db_receipt_date = receipt_metadata['unix_time']
        db_receipt_note = note

        db_receipt_total_price = to_decimal(np.sum([a['price_all'] for a in all_products]), 2)

        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO receipt (shop, date, total_price, note) VALUES (?, ?, ?, ?)
                RETURNING receipt_id
                """, (db_receipt_shop, db_receipt_date, db_receipt_total_price, db_receipt_note)
            )
            receipt_id = cur.fetchone()[0]
            db_receipt_data_path = os.path.join(self.image_save_path, str(receipt_id))

            self.conn.execute("""
            UPDATE receipt
            SET extract_data_path = ?
            WHERE receipt_id = ?;
            """, (db_receipt_data_path, receipt_id))

        for product in all_products:
            self.add_new_product_line(product, receipt_id)




if __name__ == "__main__":
    db = DatabaseService()
    with open('products.txt', 'rb') as f:
        products = pickle.load(f)
    with open('metadata.txt', 'rb') as f:
        metadata = pickle.load(f)

    dupa = [a['price_all'] for a in products['products']]
    print(np.sum(dupa))
    print(products['products'][0])

    db.add_new_receipt(products, metadata)
