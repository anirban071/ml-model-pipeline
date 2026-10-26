import pickle

from train import ModelTrainer


def test_train_saves_model_and_is_accurate(tmp_path):
    path = tmp_path / "model.pkl"
    trainer = ModelTrainer(model_path=str(path))
    _, X_test, _, y_test = trainer.generate_data()
    trainer.train_and_save()

    assert path.exists()
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model.score(X_test, y_test) > 0.80
