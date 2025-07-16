import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self, input_dim: int,
        hidden_dim: int, latent_dim: int
    ):
        super(Encoder, self).__init__()

        # 入力層サイズと隠れ層サイズと潜在層サイズを設定
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 入力層と隠れ層の定義
        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.linear_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # 活性化関数の定義
        self.relu = nn.ReLU()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        mu: torch.Tensor
            事後分布の平均 μ
        log_var: torch.Tensor
            事後分布の分散 log σ^2
        z: torch.Tensor
            潜在変数
        """
        # 事後分布の平均・分散を計算
        h = self.relu(self.linear(x))
        mu = self.linear_mu(h)
        log_var = self.linear_var(h)

        # 潜在変数を求める
        eps = torch.randn_like(torch.exp(log_var))
        z = mu + torch.exp(log_var / 2) * eps

        return mu, log_var, z


class Decoder(nn.Module):
    def __init__(
        self, input_dim: int,
        hidden_dim: int, latent_dim: int
    ):
        super(Decoder, self).__init__()

        # 出力層サイズと隠れ層サイズと潜在層サイズを設定
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 隠れ層と出力層の定義
        self.linear = nn.Linear(self.latent_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.input_dim)

        # 活性化関数の定義
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        z: torch.Tensor
            潜在変数

        Returns
        ----------
        outputs: torch.Tensor
            出力データ
        """
        # デコード
        h = self.relu(self.linear(z))
        outputs = self.sigmoid(self.linear_out(h))

        return outputs


class VAE(nn.Module):
    def __init__(
        self, input_dim: int,
        hidden_dim: int, latent_dim: int
    ):
        super(VAE, self).__init__()

        # 入力層サイズと隠れ層サイズと潜在層サイズを設定
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # エンコーダとデコーダを定義
        self.encoder = Encoder(
            self.input_dim, self.hidden_dim, self.latent_dim
        )
        self.decoder = Decoder(
            self.input_dim, self.hidden_dim, self.latent_dim
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        outputs: torch.Tensor
            出力データ
        mu: torch.Tensor
            事後分布の平均 μ
        log_var: torch.Tensor
            事後分布の分散 log σ^2
        z: torch.Tensor
            潜在変数
        """
        # エンコーダでの計算
        mu, log_var, z = self.encoder(x)

        # デコーダでの計算
        outputs = self.decoder(z)

        return outputs, mu, log_var, z
