import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

class CountryMapper:
    def __init__(self, brands: pd.Series | list[str], url: str):
        self.brands = brands
        self.url = url

        self.css_selector = ""
        self.bs = self.__load_html(
            self.__get_html()
        )

        self.__mapper__ = self.get_mapper()

    def __getitem__(self, key: str):
        return self.__mapper__.get(key.lower().replace("-", " ").replace("š", "s"), np.nan)

    def __get_html(self) -> str:
        try:
            url_request = requests.get(self.url)
            if url_request.status_code == 200:
                return url_request.content
        except Exception as e:
            print(e)

    def __load_html(self, content: str) -> BeautifulSoup:
        try:
            html_content = BeautifulSoup(content, "lxml")

            return html_content
        except Exception as e:
            print(e)
    
    def get_mapper(self):
        country_table = self.bs.find("table", {"class": "table table-bordered"})
        table_text = [i.text.lower().replace("-", " ").replace("š", "s") for i in country_table.find_all("td")]
        pass

        return dict(zip(table_text[::2], table_text[1::2]))