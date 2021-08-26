import numpy as np
import tensorflow as tf


pi = tf.constant(np.pi)
epsilon = tf.constant(1e-9)

ln = tf.math.log  # alias
lgamma = tf.math.lgamma  # alias


def nb_nll(z, model_outputs):
    μ, α = model_outputs[:, 0], model_outputs[:, 1]
    return -(
        lgamma(z + 1 / α) - lgamma(z + 1) - lgamma(1 / α)
        + (1 / α) * ln(1 / (1 + α * μ))
        + z * ln((α * μ) / (1 + α * μ))
    )


def scaler(inputs):
    mu, alpha, v = inputs[0], inputs[1], inputs[2]
    return tf.keras.layers.Concatenate()([mu * v, alpha / (2. * tf.sqrt(v))])
Scaler = tf.keras.layers.Lambda(scaler)

    
id_input = tf.keras.Input((1,), name='id')
t_input = tf.keras.Input((1,), name='t')


def STL(df, item_embeddings=None, embedding_dim=10, n_inner_layers=2, n_outer_layers=2, N=10, n_changepoints=20,
        periods=[1, 7, 365.25], T0=pd.Timestamp(0), output_activation='softplus'):
    
    T = (df.ds - T0).dt.total_seconds() / (3600 * 24)

    if item_embeddings is None:
        item_embeddings = tf.keras.layers.Reshape((embedding_dim,))(
            tf.keras.layers.Embedding(input_dim=df.id.nunique(), output_dim=embedding_dim, input_length=1)(id_input)
        )
        
    # TODO: mathematically this is unecessary. It could be that this layer helps push out
    # the embeddings, given the default weight initializer/constraints
    # If we were to stack layers, it would make more sense to do so within the seasonal embedding
    for _ in range(n_inner_layers):
        item_embeddings = tf.keras.layers.Dense(embedding_dim, activation='linear')(item_embeddings)
        
    trend_embeddings = LinearTrendEmbeddingWithMovement(input_dim=1, n_changepoints=n_changepoints, t_range=(T.min(), T.max()))([item_embeddings, t_input])

    seasonal_embeddings = [
        SeasonalEmbeddingWithMovement(period=period, N=N)([trend_embeddings, t_input])
        for period in periods
    ]

#     trend = tf.keras.layers.BatchNormalization()(
#         LinearTrendEmbeddingWithMovement(input_dim=1, n_changepoints=n_changepoints, t_range=(T.min(), T.max()))([item_embeddings, t_input])
#     )

    # perhaps a resnet layer would make sense here? If so maybe just use 4 instead
    # of `embedding_dim`
    X = tf.keras.layers.Concatenate()(seasonal_embeddings)
    for i in range(n_outer_layers):
        X = tf.keras.layers.Dense(embedding_dim, activation='linear')(X)
    X = tf.keras.layers.Dense(1, activation=output_activation)(X)

    return X, seasonal_embeddings, trend_embeddings, item_embeddings


X_μ, seasonal_embeddings_μ, trend_μ, item_embeddings_μ = STL(
    df_train,
    embedding_dim=10,
    n_inner_layers=0,
    n_outer_layers=1,
    N=10,
    output_activation='softplus')
X_α, seasonal_embeddings_α, trend_α, _ = STL(
    df_train,
    item_embeddings=item_embeddings_μ,
    embedding_dim=10,
    n_inner_layers=0,
    n_outer_layers=1,
    N=10,
    output_activation='softplus')

# model_outputs = Scaler([X_μ, X_α, ν_input])
model_outputs = tf.keras.layers.Concatenate()([X_μ, X_α])

model = tf.keras.models.Model(
    inputs=[
        id_input,
        t_input,
    ],
    outputs=model_outputs)
model.compile(loss=nb_nll, optimizer='adam')
