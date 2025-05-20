import tensorflow as tf

class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):  # ✅ Add **kwargs
        super(CBAM, self).__init__(**kwargs)            # ✅ Pass them to super()
        self.reduction_ratio = reduction_ratio


    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = tf.keras.layers.Dense(channel // self.reduction_ratio, activation='relu')
        self.shared_dense_two = tf.keras.layers.Dense(channel)
        self.spatial_conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        x = inputs * channel_attention

        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.spatial_conv(concat)
        return x * spatial_attention
