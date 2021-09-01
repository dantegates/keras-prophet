import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



def scaler(inputs):
    μ, α, ν = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    μ, α, ν = μ[:, None], α[:, None], ν[:, None]
    return tf.keras.layers.Concatenate()([μ * ν, α / (2. * tf.sqrt(ν))])
Scaler = tf.keras.layers.Lambda(scaler)


class STL(tf.keras.models.Model):
    t0 = pd.Timestamp(0)
    periods = ((1, 20), (7, 3), (365.25, 10))

    def __init__(self, *args,
                 n_items=None, embedding_dim=10, n_inner_layers=2, n_outer_layers=2,
                 trend_dim=1, n_changepoints=20, checkpoint_range=.8, t0=None, t_range=None,
                 periods=None, output_activations=('linear',),
                 n_ensemble=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_inner_layers = n_inner_layers
        self.n_outer_layers = n_outer_layers
        self.trend_dim = trend_dim
        self.n_changepoints = n_changepoints
        self.checkpoint_range = checkpoint_range
        self.t0 = t0 or self.t0
        self.t_range = t_range
        # TODO: infer proper periods from availability in data
        self.periods = periods or self.periods
        self._seasonal_periods, self._seasonal_orders = zip(*self.periods)
        self.output_activations = output_activations if isinstance(output_activations, list) else [output_activations]
        self.n_ensemble = n_ensemble

        self._init_model()

    def _init_model(self):
        # model layers
        self._id_input = tf.keras.Input((1,), name='id')
        self._t_input = tf.keras.Input((1,), name='t')

        ensemble_layers = []
        for _ in range(self.n_ensemble):
            self._trends_tensor = self._init_trends()
            self._seasonality_tensors = self._init_seasonalities()
            ensemble_layers.append(tf.keras.layers.Concatenate()(self._seasonality_tensors + [self._trends_tensor]))
        H = ensemble_layers[0] if self.n_ensemble == 1 else tf.keras.layers.Concatenate()(ensemble_layers)

        Z_θ = []
        for A in self.output_activations:
            Hi = H
            for _ in range(self.n_outer_layers):
                Hi = tf.keras.layers.Dense(1 + len(self.periods), activation='relu')(Hi)
            Z_θ.append(tf.keras.layers.Dense(1, activation=A)(Hi))


        Z_θ = Z_θ[0] if len(Z_θ) == 1 else tf.keras.layers.Concatenate()(Z_θ)

        self._model = tf.keras.models.Model(
            inputs=[self._id_input, self._t_input],
            outputs=Z_θ
        )

    def _init_trends(self):
        self._trend_weights = tf.keras.layers.Reshape((self.trend_dim, self.n_changepoints,))(
            tf.keras.layers.Embedding(input_dim=self.n_items, output_dim=self.trend_dim * self.n_changepoints, input_length=1)(self._id_input)
        )
        self._trend_k = tf.keras.layers.Reshape((self.trend_dim,))(
            tf.keras.layers.Embedding(input_dim=self.n_items, output_dim=self.trend_dim, input_length=1)(self._id_input)
        )
        self._trend_m = tf.keras.layers.Reshape((self.trend_dim,))(
            tf.keras.layers.Embedding(input_dim=self.n_items, output_dim=self.trend_dim, input_length=1)(self._id_input)
        )
        self._trend_layer = CallaborativeLinearTrendEmbedding(
            output_dim=self.trend_dim,
            t_range=self.t_range,
            n_changepoints=self.n_changepoints,
            checkpoint_range=self.checkpoint_range
        )
        return self._trend_layer([self._trend_weights, self._trend_k, self._trend_m, self._t_input])

    def _init_seasonalities(self):
        self._seasonal_weights_an = [
            self._init_item_weights(order)
            for order in self._seasonal_orders
        ]
        self._seasonal_weights_bn = [
            self._init_item_weights(order)
            for order in self._seasonal_orders
        ]

        seasonal_weights = zip(self._seasonal_weights_an, self._seasonal_weights_bn)

        self._seasonal_layers = [
            CallaborativeSeasonalEmbedding(period=p, N=N)
            for p, N in self.periods
        ]
        return [L([A_n, B_n, self._t_input]) for (A_n, B_n), L in zip(seasonal_weights, self._seasonal_layers)]

    def _init_item_weights(self, dim):
        return tf.keras.layers.Reshape((dim,))(
            tf.keras.layers.Dense(dim)(
                tf.keras.layers.Embedding(input_dim=self.n_items, output_dim=dim, input_length=1)(self._id_input)
            )
        )

    def plot(self, item_ids=None):
        item_ids = item_ids or list(range(self.n_items))

        _, axs = plt.subplots(1 + len(self._seasonal_layers), 1, figsize=(10, 10))

        inputs = [self._id_input, self._t_input]
        trend_model = tf.keras.models.Model(inputs=inputs, outputs=self._trends_tensor)
        seasonal_models = [
            tf.keras.models.Model(inputs=inputs, outputs=T)
            for T in self._seasonality_tensors
        ]

        for i in range(self.n_items):
            T = np.linspace(*self.t_range)
            axs[0].plot(self.t0 + pd.to_timedelta(T, unit='days'),
                        trend_model.predict({'t': T, 'id': np.zeros_like(T) + i}), color='C0', alpha=.5)

        for ax, seasonality_model, P in zip(axs[1:], seasonal_models, self._seasonal_periods):
            T = np.linspace(0, P)
            for i in range(self.n_items):
                seasonal_embedding = seasonality_model.predict({'t': T, 'id': np.zeros_like(T) + i})
                ax.plot(self.t0 + pd.to_timedelta(T, unit='days'),
                        seasonal_embedding, color='C0', alpha=.5)

    def sample_deltas(self):
        trend_model = tf.keras.models.Model(
            inputs=self._id_input,
            outputs=self._trend_weights)
        return trend_model.predict({'id': np.arange(self.n_items)})

    def _make_item_embeddings(self, N, D, inputs):
        E = tf.keras.layers.Reshape((D,))(
            tf.keras.layers.Embedding(input_dim=N, output_dim=D, input_length=1)(inputs)
        )
        return E


class SeasonalMixin:
    @staticmethod
    def fourier_series(a_n, b_n, n, period, t):
        x = (2 * np.pi * tf.multiply(t, n)) / period
        return (
            tf.reduce_sum(tf.multiply(tf.cos(x), a_n), axis=1, keepdims=True)
            + tf.reduce_sum(tf.multiply(tf.sin(x), b_n), axis=1, keepdims=True)
        )


class SeasonalEmbedding(tf.keras.layers.Layer, SeasonalMixin):
    def __init__(self, *args, input_dim, period, N, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.period = period
        self.N = N
        # model params
        self.a_n, self.b_n = [tf.keras.layers.Embedding(input_dim=input_dim, output_dim=N, input_length=1) for _ in range(2)]


    def call(self, input_tensor, training=False):
        id, t = input_tensor
        n = (tf.range(self.N, dtype='float') + 1)[None, :]

        instance_trend = self.fourier_series(
            tf.squeeze(self.a_n(id), axis=1),
            tf.squeeze(self.b_n(id), axis=1),
            t,
            n,
            self.period)
        global_trend = self.fourier_series(
            tf.squeeze(self.a_n(tf.zeros_like(id)), axis=1),
            tf.squeeze(self.b_n(tf.zeros_like(id)), axis=1),
            t,
            n,
            self.period)
        
        return instance_trend + global_trend
    

class CallaborativeSeasonalEmbedding(tf.keras.layers.Layer, SeasonalMixin):
    def __init__(self, *args, period, N, activation_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.N = N
        self.activation_params = activation_params or {}

        self.A_n = tf.keras.layers.Dense(N, activation='linear', **self.activation_params)
        self.B_n = tf.keras.layers.Dense(N, activation='linear', **self.activation_params)

    def call(self, input_tensor, training=False):
        A_n, B_n, t = input_tensor
        n = (tf.range(self.N, dtype='float') + 1)[None, :]
        return self.fourier_series(
            A_n, B_n,
            # self.A_n(X),
            # self.B_n(X),
            n,
            self.period,
            t)


class CallaborativeLinearTrendEmbedding(tf.keras.layers.Layer):
    def __init__(self, *args, output_dim, t_range, n_changepoints=25, checkpoint_range=.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.t_range = t_range
        self.n_changepoints = n_changepoints
        self.checkpoint_range = checkpoint_range
        # self.r = tf.keras.layers.Reshape((self.output_dim, self.n_changepoints))
        # self.δ = tf.keras.layers.Dense(self.n_changepoints * self.output_dim)
        # self.m = tf.keras.layers.Dense(self.output_dim)
        # self.k = tf.keras.layers.Dense(self.output_dim)

    def build(self, input_shape):
        self.s = tf.cast(tf.linspace(self.t_range[0], int(self.checkpoint_range * self.t_range[1]), self.n_changepoints + 1), 'float')[1:]

    def call(self, input_tensor, training=False):
        δ, m, k, t = input_tensor
        self.add_loss(tf.keras.regularizers.l1()(δ))
        return self.trend(δ, m, k, self.s, t)

    @staticmethod
    def trend(δ, m, k, s, t):
        A = tf.cast((t > s), 'float')[:, None, :]

        trend = (tf.reduce_sum(tf.multiply(A, δ), axis=-1) + k) * t

        γ = tf.multiply(-s[None, None, :], δ)
        offset = tf.reduce_sum(tf.multiply(A, γ), axis=-1) + m

        return trend + offset


class FFN(tf.keras.layers.Layer):
    def __init__(self, *args, dim, n_layers, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = [tf.keras.layers.Dense(dim, activation='relu') for _ in range(n_layers)]

    def call(self, X):
        for i, L in enumerate(self.layers):
            X = L(X)
        return X
