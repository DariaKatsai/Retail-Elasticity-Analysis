# Retail Elasticity Analysis

## Опис проекту
Цей репозиторій містить проект, що фокусується на аналізі асоціативних правил та еластичності попиту у роздрібних продажах. Проект використовує алгоритм FP-Growth для виявлення частих наборів товарів та розрахунок еластичності попиту, щоб допомогти в оптимізації стратегій продажів та маркетингу.

## Встановлення
Перед запуском проекту встановіть необхідні бібліотеки:

```python
pip install pandas numpy matplotlib mlxtend
```
## Використання
Запустити скрипт `retail_demand_elasticity_analysis.py` з командного рядка.
```python
python retail_demand_elasticity_analysis.py
```
## Функції скрипта
- `filter_data`: Фільтрація та підготовка даних.
- `get_pairs_of_goods`: Генерація пар товарів.
- `elasticity_of_demand`: Розрахунок еластичності попиту.
- `find_low_elasticity_goods`: Ідентифікація товарів з низькою еластичністю.

## Дані
Для аналізу використовується датасет 'Office-Sales.xlsx', який містить дані про роздрібні продажі.

## Результати
Результати аналізу будуть збережені у двох файлах Excel:
- `association_rules.xlsx`: містить виявлені асоціативні правила та товари з низькою еластичністю попиту.
- `inelastic_goods.xlsx`: містить детальну інформацію про товари з низькою еластичністю попиту, включаючи дані про ціни та продажі.
