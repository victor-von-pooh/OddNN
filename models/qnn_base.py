from typing import Literal, Optional

import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


class QEstimator(nn.Module):
    def __init__(
        self, qcs: list[QuantumCircuit], seed: int,
        observables: list[SparsePauliOp]
    ):
        # 親クラスのコンストラクタを呼び出す
        super(QEstimator, self).__init__()

        # 量子回路のリストを取得
        self.qc = qcs[0]
        self.feature_map = qcs[1]
        self.ansatz = qcs[2]

        # 量子層の定義
        self.qnn_est = self.set_estimator(seed, observables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            出力データ
        """
        # 量子層の順伝播計算
        out = self.qnn_est(x)

        return out

    def set_estimator(
        self, seed: int, observables: list[SparsePauliOp]
    ) -> TorchConnector:
        """
        Estimator の QNN を用意する関数

        Parameters
        ----------
        seed: int
            乱数シード値
        observables: list[SparsePauliOp]
            observable のリスト

        Returns
        ----------
        qnn: TorchConnector
            Estimator の QNN 層
        """
        # Estimator のインスタンスを用意
        estimator = Estimator(seed=seed)

        # QNN をセットアップ
        qnn_base = EstimatorQNN(
            circuit=self.qc,
            estimator=estimator,
            observables=observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters
        )
        initial_weights = 0.01 * (
            2 * algorithm_globals.random.random(qnn_base.num_weights) - 1
        )
        qnn = TorchConnector(qnn_base, initial_weights=initial_weights)

        return qnn


class QSampler(nn.Module):
    def __init__(
        self, qcs: list[QuantumCircuit], seed: int, output_dim: int
    ):
        # 親クラスのコンストラクタを呼び出す
        super(QSampler, self).__init__()

        # 量子回路のリストを取得
        self.qc = qcs[0]
        self.feature_map = qcs[1]
        self.ansatz = qcs[2]

        # 量子層の出力次元を設定
        self.output_dim = output_dim

        # 量子層の定義
        self.qnn_sam = self.set_sampler(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            出力データ
        """
        # 量子層の順伝播計算
        out = self.qnn_sam(x)

        return out

    def set_sampler(self, seed: int) -> TorchConnector:
        """
        Sampler の QNN を用意する関数

        Parameters
        ----------
        seed: int
            乱数シード値

        Returns
        ----------
        qnn: TorchConnector
            Sampler の QNN 層
        """
        # Sampler のインスタンスを用意
        sampler = Sampler(seed=seed)

        # QNN をセットアップ
        qnn_base = SamplerQNN(
            circuit=self.qc,
            sampler=sampler,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            output_shape=self.output_dim
        )
        initial_weights = 0.01 * (
            2 * algorithm_globals.random.random(qnn_base.num_weights) - 1
        )
        qnn = TorchConnector(qnn_base, initial_weights=initial_weights)

        return qnn


class QuantumNeuralNetwork(nn.Module):
    def __init__(
        self, input_dim: int, num_reuploads: int,
        net_type: Literal["Estimator", "Sampler"], seed: int,
        feature_map_reps: int = 1, ansatz_reps: int = 1,
        observables: Optional[list[SparsePauliOp]] = None,
        output_dim: Optional[int] = None
    ):
        # 親クラスのコンストラクタを呼び出す
        super(QuantumNeuralNetwork, self).__init__()

        # 入出力層の次元を設定
        self.input_dim = input_dim
        if net_type == "Sampler" and output_dim is not None:
            self.output_dim = output_dim

        # 量子回路を設計
        qcs = self.build_reuploading_circuit(
            self.input_dim, num_reuploads, feature_map_reps, ansatz_reps
        )
        self.qc = qcs[0]
        self.feature_map = qcs[1]
        self.ansatz = qcs[2]
        self.qcs = [self.qc, self.feature_map, self.ansatz]

        # observable を定義
        if net_type == "Estimator" and observables is not None:
            self.observables = observables

        # QNN を構築
        if net_type == "Estimator":
            self.qnn = QEstimator(self.qcs, seed, self.observables)
        elif net_type == "Sampler":
            self.qnn = QSampler(self.qcs, seed, self.output_dim)
        else:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        x: torch.Tensor
            入力データ

        Returns
        ----------
        out: torch.Tensor
            出力データ
        """
        # 量子層の順伝播計算
        out = self.qnn(x)

        return out

    def build_reuploading_circuit(
        self, num_qubits: int, num_reuploads: int,
        feature_map_reps: int = 1, ansatz_reps: int = 1
    ) -> list[QuantumCircuit]:
        """
        QNN 用の量子回路を設計する関数

        Parameters
        ----------
        num_qubits: int
            量子ビット数
        num_reuploads: int
            Re-Upload の回数
        feature_map_reps: int = 1
            特徴マップのリピート数
        ansatz_reps: int = 1
            ansatz のリピート数

        Returns
        ----------
        out: list[QuantumCircuit]
            量子回路のリスト
        """
        # 量子回路を初期化
        qc = QuantumCircuit(num_qubits)

        # 特徴マップと ansatz の定義
        feature_map = zz_feature_map(num_qubits, reps=feature_map_reps)
        ansatz = real_amplitudes(
            num_qubits, entanglement="linear", reps=ansatz_reps
        )

        # Re-Upload
        for _ in range(num_reuploads):
            qc.compose(feature_map, inplace=True)
            qc.compose(ansatz, inplace=True)

        # 量子回路をまとめてリスト化
        out = [qc, feature_map, ansatz]

        return out
