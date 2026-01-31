from typing import Optional

import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LSTMCell, self).__init__()

        # 入力層サイズと隠れ層サイズを設定
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 各種重みとバイアスの定義
        self.weight_ih = nn.Parameter(
            torch.randn(4 * self.hidden_dim, self.input_dim)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(4 * self.hidden_dim, self.hidden_dim)
        )
        self.bias_ih = nn.Parameter(torch.zeros(4 * self.hidden_dim))
        self.bias_hh = nn.Parameter(torch.zeros(4 * self.hidden_dim))

        # 活性化関数の定義
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x_t: torch.Tensor
            入力データ
        h_prev: torch.Tensor
            1ステップ前の隠れ状態
        c_prev: torch.Tensor
            1ステップ前のセル

        Returns
        ----------
        h_t: torch.Tensor
            隠れ状態
        c_t: torch.Tensor
            セル
        """
        # 総合的なゲート計算
        gates = (
            torch.matmul(x_t, self.weight_ih.t()) +
            torch.matmul(h_prev, self.weight_hh.t()) +
            self.bias_ih + self.bias_hh
        )

        # ゲート分割
        i, f, g, o = gates.chunk(4, dim=1)

        # 活性化関数適用
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)

        # セル・隠れ状態の更新
        c_t = f * c_prev + i * g
        h_t = o * self.tanh(c_t)

        return h_t, c_t


class LSTMLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int,
        bidirectional: bool = False, residual: bool = False
    ):
        super(LSTMLayer, self).__init__()

        # 入力層サイズと隠れ層サイズを設定
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 双方向にするかを設定
        self.bidirectional = bidirectional

        # 残差接続するかを設定
        self.residual = residual

        # (双方向の)LSTMセルのインスタンスの作成
        self.cell_fwd = LSTMCell(self.input_dim, self.hidden_dim)
        if self.bidirectional:
            self.cell_bwd = LSTMCell(self.input_dim, self.hidden_dim)

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ
        h_0: torch.Tensor
            初期化された隠れ状態
        c_0: torch.Tensor
            初期化されたセル

        Returns
        ----------
        outputs: torch.Tensor
            出力データ
        h_t: torch.Tensor
            隠れ状態
        c_t: torch.Tensor
            セル
        """
        # 入力長を取得
        seq_len, _, _ = x.size()

        # forward方向の処理
        h_fwd, c_fwd = h_0[0], c_0[0]
        outputs_fwd = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.cell_fwd(x[t], h_fwd, c_fwd)
            outputs_fwd.append(h_fwd.unsqueeze(0))
        outputs_fwd = torch.cat(outputs_fwd, dim=0)

        # 双方向ではない場合用に分岐
        if not self.bidirectional:
            outputs = outputs_fwd
            h_t = h_fwd.unsqueeze(0)
            c_t = c_fwd.unsqueeze(0)
        else:
            # backward方向の処理
            h_bwd, c_bwd = h_0[1], c_0[1]
            outputs_bwd = []
            for t in reversed(range(seq_len)):
                h_bwd, c_bwd = self.cell_bwd(x[t], h_bwd, c_bwd)
                outputs_bwd.insert(0, h_bwd.unsqueeze(0))
            outputs_bwd = torch.cat(outputs_bwd, dim=0)

            # forward/backwardの出力を連結
            outputs = torch.cat([outputs_fwd, outputs_bwd], dim=2)
            h_t = torch.stack([h_fwd, h_bwd], dim=0)
            c_t = torch.stack([c_fwd, c_bwd], dim=0)

        # 残差接続
        if self.residual and outputs.shape == x.shape:
            outputs = outputs + x

        return outputs, h_t, c_t


class LSTM(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 1,
        bidirectional: bool = False, residual: bool = False
    ):
        super(LSTM, self).__init__()

        # 入力層サイズと隠れ層サイズを設定
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 層数を設定
        self.num_layers = num_layers

        # 双方向にするかを設定
        self.bidirectional = bidirectional

        # 残差接続するかを設定
        self.residual = residual

        # LSTM全体の構造を作成する
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            in_dim = self.input_dim if layer == 0 else self.hidden_dim * (
                2 if self.bidirectional else 1
            )
            self.layers.append(
                LSTMLayer(
                    in_dim, self.hidden_dim,
                    self.bidirectional, self.residual
                )
            )

    def forward(
        self, x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        c_0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ
        h_0: Optional[torch.Tensor]=None
            初期化された隠れ状態
        c_0: Optional[torch.Tensor]=None
            初期化されたセル

        Returns
        ----------
        outputs: torch.Tensor
            出力データ
        h_t: torch.Tensor
            隠れ状態
        c_t: torch.Tensor
            セル
        """
        # バッチサイズを取得
        _, batch_size, _ = x.size()

        # 方向数を指定
        num_directions = 2 if self.bidirectional else 1

        # 初期状態の隠れ状態とセルを定義
        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * num_directions, batch_size,
                self.hidden_dim, device=x.device
            )
        if c_0 is None:
            c_0 = torch.zeros(
                self.num_layers * num_directions, batch_size,
                self.hidden_dim, device=x.device
            )

        # スタックの用意
        h_n = []
        c_n = []
        layer_input = x

        # 層ごとに処理・格納
        for layer_idx, layer in enumerate(self.layers):
            # 初期化
            h_start = layer_idx * num_directions
            h_i = h_0[h_start:h_start + num_directions]
            c_i = c_0[h_start:h_start + num_directions]

            # 次の層に更新
            layer_output, h_i_out, c_i_out = layer(layer_input, h_i, c_i)
            layer_input = layer_output

            # スタックに格納
            h_n.append(h_i_out)
            c_n.append(c_i_out)

        # 最終層の出力
        outputs = layer_output
        h_t = torch.cat(h_n, dim=0)
        c_t = torch.cat(c_n, dim=0)

        return outputs, (h_t, c_t)
