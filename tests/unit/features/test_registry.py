from phinance.features.registry import FeatureRegistry


def test_registry_save_load_and_rank(tmp_path):
    path = tmp_path / "registry.json"
    registry = FeatureRegistry.open(path)

    registry.set_autoencoder("checkpoints/model.npz", latent_dim=8, metrics={"reconstruction_loss": 0.1})
    registry.add_gp_feature("add(close, volume)", metrics={"fitness": 0.4})
    registry.add_gp_feature("sub(close, open)", metrics={"fitness": 0.2})
    registry.save()

    reloaded = FeatureRegistry.open(path)
    assert reloaded.payload["autoencoder"]["latent_dim"] == 8
    top = reloaded.top_gp_features(limit=1)
    assert top[0]["expression"] == "add(close, volume)"
