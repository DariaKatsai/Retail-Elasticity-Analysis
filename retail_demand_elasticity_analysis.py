"""
retail_demand_elasticity_analysis.py

Опис: Цей скрипт виконує комплексний аналіз роздрібних даних, включаючи фільтрацію та обробку даних, аналіз асоціативних правил з використанням алгоритму FP-Growth та розрахунок еластичності попиту на товари. Основні функції включають виявлення частих наборів товарів, аналіз взаємозалежностей між товарами за допомогою метрик підтримки, підйому та впевненості, а також ідентифікацію товарів з низькою еластичністю попиту для оптимізації стратегій ціноутворення та продажів.

Автор: Кацай Дар'я
Дата створення: 1 серпня 2023 року
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

# відфільтрувати дані
def filter_data(fname, unit_price, quantity, shipping_cost, discount, product, date, invoice):
    """
    Фільтрація та підготовка даних з Excel файлу.
    Параметри: шлях до файлу, назви стовпців для фільтрації.
    """
    # read data with excel file
    online_retail = pd.read_excel(fname)
    
    # filter data
    online_retail = online_retail[(online_retail[unit_price] > 0) 
                                  & (online_retail[quantity] > 0)
                                  & (online_retail[quantity] < 1000)
                                  & (online_retail[shipping_cost] > 0)]
    
    # delete duplicates and empty columns
    online_retail = online_retail.drop_duplicates()
    online_retail = online_retail.dropna(subset=online_retail.columns.difference([discount]))
    
    # convert date column to datetime type
    online_retail[date] = pd.to_datetime(online_retail[date])
    
    # sort by year in ascending order
    online_retail = online_retail.sort_values(by=date)
    
    # replace ','
    online_retail[product] = online_retail[product].str.replace(',', ' -')
    
    # data grouping in DataFrameGroupBy
    grouped_transactions = online_retail[[invoice, date, product, unit_price, quantity]]

    # save results in excel file
    #grouped_transactions.to_excel('grouped_data.xlsx', index=False)
    
    return grouped_transactions

# отрмати дані після використання метрик
def get_pairs_of_goods(dataset, product_name):
    """
    Генерація пар товарів за допомогою FP-Growth.
    Параметри: датасет, назва стовпця з назвою продукту.
    """
    # group by check numbers
    dataframe = dataset.groupby(dataset.columns[0])[product_name].apply(list)
    
    te = TransactionEncoder()
    te_ary = te.fit(dataframe).transform(dataframe)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # find frequent itemsets with minimum support
    patterns = fpgrowth(df, min_support=0.0001, use_colnames=True)

    rules_confidence = association_rules(patterns, metric="confidence", min_threshold=0.1)
    rules_lift = association_rules(patterns, metric="lift", min_threshold=100)
    
    # filter by lift
    filtered_rules = rules_lift[rules_lift['confidence'].isin(rules_confidence['confidence'])]
    output_data = filtered_rules[['antecedents', 'consequents']].copy()
    
    # convert in str
    output_data['antecedents'] = output_data['antecedents'].apply(lambda x: ', '.join(list(x)))
    output_data['consequents'] = output_data['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # new column 'sorted_pair'
    output_data['sorted_pair'] = output_data.apply(lambda row: frozenset(sorted([row['antecedents'], row['consequents']])), axis=1)

    # sort by 'sorted_pair' 
    output_data.drop_duplicates(subset='sorted_pair', keep='first', inplace=True)

    # delete column 'sorted_pair'
    output_data.drop(columns='sorted_pair', inplace=True)

    #output_data.to_excel("ffff.xlsx", index=False)
    return output_data
 
# еластичність 
def elasticity_of_demand(quantity2, quantity1, price2, price1):
    """
    Розрахунок еластичності попиту.
    Параметри: кількість і ціна товару в два різні періоди.
    """
    
    percentage_quantity = (quantity2 - quantity1) / quantity1
    percentage_price = (price2 - price1) / price1

    if percentage_price == 0:
        return 0  
    else:
        elasticity = percentage_quantity / percentage_price
        if np.isinf(elasticity):  # Перевірка на нескінченність
            return -1 
        else:
            return abs(round(elasticity))
   


def add_dataframe(data_df, name_product, price):
    
    years = [2017, 2018, 2019, 2020]
    
    if name_product not in data_df.index:
        data_df.loc[name_product] = name_product
        
    for year in years:
        if year in price:
            data_df.loc[name_product, year] = round(price[year], 2)
        else:
            data_df.loc[name_product, year] = 0
    
    return data_df


def get_price_quantity_by_years(table, product_name, product, date, unit_price, quantity, data):
    product_data_copy = table[table[product_name] == product]
    product_data = product_data_copy.copy()
    product_data.loc[:, date] = pd.to_datetime(product_data[date]).dt.year

    # Знаходимо мінімальний та максимальний рік
    min_year = product_data[date].min()
    max_year = product_data[date].max()
    
    average_by_year = product_data.groupby(date)[unit_price].max()

    # Обчислюємо середню вартість за мінімальний та максимальний рік
    average_by_min_year = round(average_by_year.loc[min_year], 2)
    average_by_max_year = round(average_by_year.loc[max_year], 2)
    
    number_quantity = product_data.groupby(date)[quantity].sum()
     # Обчислюємо кількість товарів за мінімальний та максимальний рік
    quantity_by_min_year = number_quantity.loc[min_year]
    quantity_by_max_year = number_quantity.loc[max_year]
    
    data = add_dataframe(data, product, average_by_year)
    product_elasticity = elasticity_of_demand(quantity_by_max_year, 
                                              quantity_by_min_year, 
                                              average_by_max_year, 
                                              average_by_min_year)
    return product_elasticity    


def find_low_elasticity_goods(table, df_pairs_goods, product_name, unit_price, quantity, date):
    """
    Ідентифікація товарів з низькою еластичністю попиту.
    Параметри: датасет, результати FP-Growth, назви стовпців.
    """
    list_tuples = [(row['antecedents'], row['consequents']) for index, row in df_pairs_goods.iterrows()]
    
    columns = ['Product'] + [2017, 2018, 2019, 2020]
    data = pd.DataFrame(columns=columns)
    
    df_pairs_goods['Selected Product'] = None
    df_pairs_goods['elacity'] = None
    
    for tup in list_tuples:
        product1, product2 = tup
        # find data for product1
        elasticity1 = get_price_quantity_by_years(table, product_name, product1, 
                                                  date, unit_price, quantity, data)
        elasticity2 = get_price_quantity_by_years(table, product_name, product2, 
                                                  date, unit_price, quantity, data)
       #print(product1 + "=====" + str(elasticity1))
        #print(product2 + "=====" + str(elasticity2))
        print(">>>>>>>>>>>>>>>")
        i = 0
        if (elasticity1 > elasticity2 and elasticity2 != -1) or elasticity2 == 0:
            i = 2
        elif (elasticity1 < elasticity2 and elasticity1 != -1) or elasticity1 == 0:
            i = 1

        for index, row in df_pairs_goods.iterrows():
            if product1 in row['antecedents'] and product2 in row['consequents']:
                if i == 1:
                    df_pairs_goods.loc[index, 'Selected Product'] = product1
                    df_pairs_goods.loc[index, 'elacity'] = elasticity1
            
                elif i == 2:
                    df_pairs_goods.loc[index, 'Selected Product'] = product2
                    df_pairs_goods.loc[index, 'elacity'] = elasticity2
                    
    df_pairs_goods = df_pairs_goods.dropna()
    df_pairs_goods.to_excel("association_rules.xlsx", index=False)
    
    products = df_pairs_goods['Selected Product'].drop_duplicates()
    sheet2_filtered = data[data['Product'].isin(products)]
    with pd.ExcelWriter("association_rules.xlsx", engine='openpyxl', mode='a') as writer:
        sheet2_filtered.to_excel(writer, sheet_name="Sheet2", index=False)


def get_list_elasity_product(table, product_name, product, unit_price, quantity):
    product_data = table[table[product_name] == product]
    elasticities = []
    for i in range(len(product_data) - 1):
        elasticity = elasticity_of_demand(product_data[quantity].iloc[i+1], product_data[quantity].iloc[i], 
                                          product_data[unit_price].iloc[i+1], product_data[unit_price].iloc[i])
        elasticities.append(elasticity)
    return elasticities

def graphic_image(table, product_name, product, elasticity, date):
    product_data = table[table[product_name] == product]
    
    save_data_in_file("inelastic_goods1.xlsx", product_data, product)
    
    plt.figure(figsize=(10, 6))
    plt.plot(product_data[date][:len(elasticity)], elasticity)
    plt.xlabel("Дата")
    plt.ylabel("Коефіцієнт еластичності")
    plt.title(product)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def save_data_in_file(name_file, df, name_sheet):
    with pd.ExcelWriter(name_file, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name=name_sheet, index=False)

def get_table(table, df_pairs_goods, product_name, unit_price, quantity, date):
    list_tuples = [(row['antecedents'], row['consequents']) for index, row in df_pairs_goods.iterrows()]
    df_pairs_goods['Selected Product'] = None
    
    for tup in list_tuples:
        product1, product2 = tup
        
        # find data for product1
        elasticity1 = get_list_elasity_product(table, product_name, product1, unit_price, quantity)
        elasticity2 = get_list_elasity_product(table, product_name, product2, unit_price, quantity)

        average1 = np.mean(elasticity1)
        average2 = np.mean(elasticity2)


        for index, row in df_pairs_goods.iterrows():
            if product1 in row['antecedents'] and product2 in row['consequents']:
                if average1 < average2:
                    df_pairs_goods.loc[index, 'Selected Product'] = product1
                    graphic_image(table, product_name, product1, elasticity1, date)
                elif average1 > average2:
                    df_pairs_goods.loc[index, 'Selected Product'] = product2
                    graphic_image(table, product_name, product2, elasticity2, date)
                    
    df_pairs_goods = df_pairs_goods.dropna()
    save_data_in_file("inelastic_goods1.xlsx", df_pairs_goods, "Products")
    #df_pairs_goods.to_excel("inelastic_goods.xlsx", index=False)


    
input_data = filter_data("Office-Sales.xlsx", "UnitPrice", 
            "OrderQuantity", "ShippingCost", "Discount %", 
            "ProductName", "OrderDate", "SalesOrderNumber")

rules = get_pairs_of_goods(input_data, "ProductName")
get_table(input_data, rules, "ProductName", "UnitPrice", "OrderQuantity", "OrderDate")

#find_low_elasticity_goods(input_data, rules, "ProductName", "UnitPrice", "OrderQuantity", "OrderDate")

