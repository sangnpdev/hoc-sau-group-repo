"""Kiểm tra nhanh dữ liệu; chạy từ thư mục Tuan03/Lab03."""

import pandas as pd

from house_price_pipeline import TARGET, load_competition_data


def main():
    # In kích thước, kiểu dữ liệu, giá trị thiếu và thống kê của SalePrice.
    train, test = load_competition_data("data")
    print(f"Train: {train.shape[0]} rows x {train.shape[1]} columns")
    print(f"Test:  {test.shape[0]} rows x {test.shape[1]} columns")
    print("\nTrain sample:")
    print(train.head().to_string(index=False))
    print("\nColumn data types (pandas):")
    print(train.dtypes.value_counts().to_string())
    print("\nColumn inventory: pandas dtype, missing count, example:")
    inventory = pd.DataFrame(
        {
            "dtype": train.dtypes.astype(str),
            "missing": train.isna().sum(),
            "example": [
                train[column].dropna().iloc[0] if train[column].notna().any() else "<all missing>"
                for column in train.columns
            ],
        }
    )
    # Bảng inventory giúp nhìn đồng thời kiểu dữ liệu, số ô thiếu và một ví dụ.
    print(inventory.to_string())
    print("\nMissing values in train (top 20):")
    missing = train.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(20).to_string())
    print("\nSalePrice summary:")
    print(train[TARGET].describe().to_string())
    print("\nTarget absent from test:", TARGET not in test.columns)


if __name__ == "__main__":
    main()
