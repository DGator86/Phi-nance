import numpy as np

from phinance.features.autoencoder import AutoencoderConfig, MarketAutoencoder


def test_autoencoder_train_and_encode_shape():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 20)).astype(np.float32)

    model = MarketAutoencoder(
        AutoencoderConfig(input_dim=20, latent_dim=6, hidden_dim=16, epochs=5, batch_size=32, learning_rate=1e-2)
    )
    history = model.fit(x)

    assert len(history["reconstruction_loss"]) == 5
    assert history["reconstruction_loss"][-1] <= history["reconstruction_loss"][0]

    z = model.encode(x[:4])
    assert z.shape == (4, 6)


def test_autoencoder_save_and_load(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 12)).astype(np.float32)

    model = MarketAutoencoder(AutoencoderConfig(input_dim=12, latent_dim=4, epochs=2))
    model.fit(x)
    ckpt = model.save(tmp_path / "ae.npz")

    loaded = MarketAutoencoder.load(ckpt)
    z = loaded.encode(x[:3])
    assert z.shape == (3, 4)
