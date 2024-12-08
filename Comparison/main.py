from datasets import load_breast_cancer_dataset, load_wine_dataset, generate_custom_dataset, load_digits_dataset, load_iris_dataset
from sklearn.preprocessing import MinMaxScaler
import mpc_vs_dpc
import mbscan_vs_dbscan  # Assurez-vous que ce module est correctement importé


def main():
    print("Choisissez une comparaison:")
    print("1. Comparer MBSCAN et DBSCAN")
    print("2. Comparer MPC et DPC (CFSFDP)")
    comparison_choice = input("Entrez votre choix (1 ou 2): ")

    print("\nChoisissez un dataset:")
    print("1. Breast Cancer Wisconsin")
    print("2. Wine")
    print("3. Custom dataset")
    print("4. Digits")
    print("5. Iris")
    dataset_choice = input("Entrez votre choix (1, 2, 3, 4 ou 5): ")

    wbcd_param = False
    wine_param = False
    custom_param = False
    digits_param = False
    iris_param = False

    if dataset_choice == '1':
        X_orig, y = load_breast_cancer_dataset(normalize=False)
        wbcd_param = True
    elif dataset_choice == '2':
        X_orig, y = load_wine_dataset(normalize=False)
        wine_param = True
    elif dataset_choice == '3':
        X_orig, y = generate_custom_dataset(normalize=False)
        custom_param = True
    elif dataset_choice == '4':
        X_orig, y = load_digits_dataset(normalize=False)
        digits_param = True
    elif dataset_choice == '5':
        X_orig, y = load_iris_dataset(normalize=False)
        iris_param = True
    else:
        print("Choix de dataset invalide.")
        return

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_orig)

    display_heatmaps_input = input("\nVoulez-vous afficher les heatmaps des matrices de distances ? (o/n) : ")
    display_heatmaps_flag = display_heatmaps_input.lower() in ['o', 'oui', 'y', 'yes']

    display_matrices_input = input("\nVoulez-vous afficher les matrices de distances dans la console ? (o/n) : ")
    display_matrices_flag = display_matrices_input.lower() in ['o', 'oui', 'y', 'yes']

    if comparison_choice == '1':
        mbscan_vs_dbscan.run_mbscan_vs_dbscan(
            X_normalized,
            X_orig,
            y,
            display_heatmaps_flag,
            display_matrices_flag,
            wbcd_param=wbcd_param,
            wine_param=wine_param,
            custom_param=custom_param,
            digits_param=digits_param,
            iris_param=iris_param
        )
    elif comparison_choice == '2':
        mpc_vs_dpc.run_mpc_vs_dpc(
            X_normalized,
            X_orig,
            y,
            display_heatmaps_flag,
            display_matrices_flag,
            wbcd_param=wbcd_param,
            wine_param=wine_param,
            custom_param=custom_param,
            digits_param=digits_param,
            iris_param=iris_param
        )
    else:
        print("Choix de comparaison invalide.")


if __name__ == '__main__':
    main()
