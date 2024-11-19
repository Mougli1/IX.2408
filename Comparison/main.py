
from datasets import load_iris_dataset, load_wine_dataset, generate_blobs_dataset, load_custom_dataset, load_matlab_dataset

def main():
    print("Choisissez une comparaison:")
    print("1. Comparer MBSCAN et DBSCAN")
    print("2. Comparer MPC et DPC (CFSFDP)")
    comparison_choice = input("Entrez votre choix (1 ou 2): ")

    print("\nChoisissez un dataset:")
    print("1. Iris")
    print("2. Wine")
    print("3. Blobs")
    print("4. Dataset personnalisé")
    print("5. Dataset MATLAB (data.mat)")
    dataset_choice = input("Entrez votre choix (1, 2, 3, 4 ou 5): ")

    if dataset_choice == '1':
        X, y = load_iris_dataset()
    elif dataset_choice == '2':
        X, y = load_wine_dataset()
    elif dataset_choice == '3':
        X, y = generate_blobs_dataset()
    elif dataset_choice == '4':
        X, y = load_custom_dataset()
    elif dataset_choice == '5':
        X, y = load_matlab_dataset()
        if X is None:
            print("Impossible de charger le dataset MATLAB. Veuillez vérifier le fichier.")
            return
    else:
        print("Choix de dataset invalide.")
        return
    display_heatmaps_input = input("\nVoulez-vous afficher les heatmaps des matrices de distances ? (o/n) : ")
    if display_heatmaps_input.lower() in ['o', 'oui', 'y', 'yes']:
        display_heatmaps_flag = True
    else:
        display_heatmaps_flag = False
    display_matrices_input = input("\nVoulez-vous afficher les matrices de distances dans la console ? (o/n) : ")
    if display_matrices_input.lower() in ['o', 'oui', 'y', 'yes']:
        display_matrices_flag = True
    else:
        display_matrices_flag = False

    if comparison_choice == '1':
        import mbscan_vs_dbscan
        mbscan_vs_dbscan.run_mbscan_vs_dbscan(X, y, display_heatmaps_flag, display_matrices_flag)
    elif comparison_choice == '2':
        import mpc_vs_dpc
        mpc_vs_dpc.run_mpc_vs_dpc(X, y, display_heatmaps_flag, display_matrices_flag)
    else:
        print("Choix de comparaison invalide.")

if __name__ == '__main__':
    main()
