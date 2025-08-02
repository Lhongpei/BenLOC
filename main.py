import argparse
from BenLOC import BenLOC, TabParams, TabNetRegressor
from sklearn.ensemble import RandomForestRegressor
from BenLOC.DL.gnn_pairwise.train_gnn_predictor import gnn_pairwise
from BenLOC.DL.gnn_predictor.trainPyG.predictTrain import gnn_predictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Program to run BenLOC")

    # add choice: tab_ml, tab_dl, gnn_pairwise, gnn_predictor
    # parser, choice:
    parser.add_argument(
        "--mode",
        dest="mode",
        choices=["tab_ml", "tab_dl", "gnn_pairwise", "gnn_predictor"],
        default="tab_ml",
        help="Mode of training"
    )

    args = parser.parse_args()

    if args.mode == "tab_ml":
        # Tabular ML
        params = TabParams(default="default", label_type="log_scaled", shift_scale=10)
        model = BenLOC(params)
        model.load_dataset("indset", processed=True)
        model.set_trainner(RandomForestRegressor())
        model.train_test_split_by_splitfile('./BenLOC/data/fold_name/indset_fold_2.pkl')
        model.fit()
        evaluation_results = model.evaluate()
        print(evaluation_results)

    elif args.mode == "tab_dl":
        # Tabular DL
        params = TabParams(default="default", label_type="log_scaled", shift_scale=10)
        model = BenLOC(params)
        model.load_dataset("indset", processed=True)
        tabmodel = TabNetRegressor(n_d=5, n_a=5)
        tabmodel.fit(model.get_X, model.get_Y, max_epochs=1)
        model.set_trainner(tabmodel.network)
        model.train_test_split_by_splitfile('./BenLOC/data/fold_name/indset_fold_2.pkl')
        model.fit()
        evaluation_results = model.evaluate()
        print(evaluation_results)

    elif args.mode == "gnn_pairwise":
        # Pairwise GNN
        gnn_pairwise()

    elif args.mode == "gnn_predictor":
        # Predictor GNN
        gnn_predictor()

    else:
        raise ValueError("Invalid mode selected. Choose from 'tab_ml', 'tab_dl', 'gnn_pairwise', or 'gnn_predictor'.")
