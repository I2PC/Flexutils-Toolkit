import tensorflow as tf


class CBAM(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, kernel_size=7, **kwargs):
        """
        Args:
            reduction_ratio: Reduction ratio for the channel attention module.
            kernel_size: Kernel size for the spatial attention convolution.
        """
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel = input_shape[-1]
        # Ensure reduction channels is at least 1
        reduction_channels = max(1, channel // self.reduction_ratio)

        # Shared MLP for channel attention
        self.shared_dense_one = tf.keras.layers.Dense(
            reduction_channels,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.shared_dense_two = tf.keras.layers.Dense(
            channel,
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )

        # Convolution layer for spatial attention
        self.conv_spatial = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        # ----- Channel Attention Module -----
        # Compute channel-wise statistics: global average and max pooling
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2])  # shape: (B, C)
        max_pool = tf.reduce_max(inputs, axis=[1, 2])  # shape: (B, C)

        # Shared MLP (applied to both average and max pooled features)
        mlp_avg = self.shared_dense_two(self.shared_dense_one(avg_pool))
        mlp_max = self.shared_dense_two(self.shared_dense_one(max_pool))
        # Combine and apply sigmoid activation to get the channel attention weights
        channel_attention = tf.nn.sigmoid(mlp_avg + mlp_max)
        # Reshape to (B, 1, 1, C) for broadcasting
        channel_attention = tf.reshape(channel_attention, [-1, 1, 1, inputs.shape[-1]])
        # Refine the input features with channel attention
        x = inputs * channel_attention

        # ----- Spatial Attention Module -----
        # Compute spatial statistics: average and max pooling along channel axis
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)  # shape: (B, M, M, 1)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)  # shape: (B, M, M, 1)
        # Concatenate along the channel axis and convolve
        concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)  # shape: (B, M, M, 2)
        spatial_attention = self.conv_spatial(concat)  # shape: (B, M, M, 1)
        # Apply spatial attention
        x = x * spatial_attention

        return x

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size,
        })
        return config


# class CBAM(tf.keras.layers.Layer):
#     """
#     Convolutional Block Attention Module (CBAM) layer for TensorFlow/Keras.
#
#     This layer implements the CBAM attention mechanism as described in
#     "CBAM: Convolutional Block Attention Module" (Woo et al., 2018).  It applies
#     both channel and spatial attention to the input feature map.
#
#     Args:
#         reduction_ratio (int):  The reduction ratio for the channel attention module.
#             Defaults to 16.  Must be a positive integer. A smaller ratio means *more*
#             channel compression (stronger attention).  A value of 1 means no reduction
#             (the number of channels in the intermediate representation is the same as the
#             input). Good values are usually in the range [8, 16, 32, ...]
#         name (str, optional):  The name of the layer. Defaults to None.
#         kernel_size (int or tuple): The size of the convolutional kernel used in
#             the spatial attention module. Defaults to 7.  Can be an integer (same
#             kernel size in all spatial dimensions) or a tuple of integers (specifying
#             the kernel size for each spatial dimension).
#         activation: The activation function used in the channel attention MLP. Defaults to ReLU.
#         use_bias: Whether to use bias in the channel attention dense layers. Defaults to False.
#         kernel_initializer: The kernel initializer used in the channel attention dense layers.
#              Defaults to Glorot Uniform.
#         bias_initializer: The bias initializer used in the channel attention dense layers. Defaults to Zeros.
#
#     Input Shape:
#         (batch_size, height, width, channels) -  4D tensor.
#
#     Output Shape:
#         (batch_size, height, width, channels) - 4D tensor with the same shape as the input.
#
#     Raises:
#         ValueError: if `reduction_ratio` is not a positive integer.
#     """
#
#     def __init__(self, reduction_ratio=16, name=None, kernel_size=7, activation='relu',
#                  use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
#         super(CBAM, self).__init__(name=name, **kwargs)
#
#         if not isinstance(reduction_ratio, int) or reduction_ratio <= 0:
#             raise ValueError("`reduction_ratio` must be a positive integer.")
#
#         self.reduction_ratio = reduction_ratio
#         self.kernel_size = kernel_size
#         self.activation = activation
#         self.use_bias = use_bias
#         self.kernel_initializer = kernel_initializer
#         self.bias_initializer = bias_initializer
#
#     def build(self, input_shape):
#         input_shape = tf.TensorShape(input_shape)
#         channel_axis = -1  # Assuming channels_last data format
#         num_channels = input_shape[channel_axis]
#         if num_channels is None:
#             raise ValueError("The channel dimension of the input tensor must be defined.")
#
#         # Channel Attention Module
#         self.gap = tf.keras.layers.GlobalAveragePooling2D()
#         self.gmp = tf.keras.layers.GlobalMaxPooling2D()
#
#         self.dense1 = tf.keras.layers.Dense(
#             num_channels // self.reduction_ratio,
#             activation=self.activation,
#             use_bias=self.use_bias,
#             kernel_initializer=self.kernel_initializer,
#             bias_initializer=self.bias_initializer
#             )
#         self.dense2 = tf.keras.layers.Dense(
#             num_channels,
#             use_bias=self.use_bias,
#             kernel_initializer=self.kernel_initializer,
#             bias_initializer=self.bias_initializer
#         )
#
#         # Spatial Attention Module
#         self.conv = tf.keras.layers.Conv2D(
#             filters=1,
#             kernel_size=self.kernel_size,
#             padding='same',
#             activation='sigmoid',
#             kernel_initializer='glorot_uniform',
#             use_bias=False,  # As per the original paper, no bias in spatial conv
#             )
#
#         super(CBAM, self).build(input_shape)
#
#
#     def call(self, inputs):
#         # Channel Attention
#         gap_output = self.dense2(self.dense1(self.gap(inputs)))  # (B, C)
#         gmp_output = self.dense2(self.dense1(self.gmp(inputs)))  # (B, C)
#         channel_attention = tf.keras.activations.sigmoid(gap_output + gmp_output)  # (B, C)
#         # Reshape channel_attention to (B, 1, 1, C) for broadcasting
#         channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, axis=1), axis=1) # (B, 1, 1, C)
#         x = inputs * channel_attention # (B, M, M, C)
#
#         # Spatial Attention
#         avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True) # (B, M, M, 1)
#         max_pool = tf.reduce_max(x, axis=-1, keepdims=True)  # (B, M, M, 1)
#         concatenated = tf.concat([avg_pool, max_pool], axis=-1) # (B, M, M, 2)
#         spatial_attention = self.conv(concatenated)   # (B, M, M, 1)
#         x = x * spatial_attention # (B, M, M, C)
#
#         return x
#
#     def get_config(self):
#         config = super(CBAM, self).get_config()
#         config.update({
#             'reduction_ratio': self.reduction_ratio,
#             'kernel_size': self.kernel_size,
#             'activation': self.activation,
#             'use_bias': self.use_bias,
#             'kernel_initializer': self.kernel_initializer,
#             'bias_initializer': self.bias_initializer
#         })
#         return config


class EfficientChannelAttention(tf.keras.layers.Layer):
    """
    Efficient Channel Attention (ECA) Module.
    Applies global average pooling followed by a 1D convolution along the channel dimension.
    """
    def __init__(self, kernel_size=3, **kwargs):
        """
        Args:
            kernel_size (int): Kernel size for the 1D convolution.
        """
        super(EfficientChannelAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # input_shape: (B, M, M, C)
        self.num_channels = input_shape[-1]
        # Conv1D expects input shape (B, steps, channels).
        # We'll reshape our channel descriptor (B, C) -> (B, C, 1)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        super(EfficientChannelAttention, self).build(input_shape)

    def call(self, x):
        # Global average pooling over spatial dimensions: (B, M, M, C) -> (B, C)
        avg_pool = tf.reduce_mean(x, axis=[1, 2])
        # Reshape to (B, C, 1) to use Conv1D along the channel dimension
        avg_pool = tf.expand_dims(avg_pool, axis=-1)
        # Apply 1D convolution: result shape is (B, C, 1)
        attn = self.conv1d(avg_pool)
        # Remove the last singleton dimension: (B, C)
        attn = tf.squeeze(attn, axis=-1)
        # Apply sigmoid to get channel attention weights in [0, 1]
        attn = tf.sigmoid(attn)
        # Reshape to (B, 1, 1, C) for broadcasting and multiply with the input
        attn = tf.reshape(attn, [-1, 1, 1, self.num_channels])
        return x * attn

    def get_config(self):
        config = super(EfficientChannelAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config


class SpatialAttention(tf.keras.layers.Layer):
    """
    Spatial Attention Module.
    Uses average and max pooling along the channel axis followed by a convolution to generate
    a spatial attention map.
    """
    def __init__(self, kernel_size=7, **kwargs):
        """
        Args:
            kernel_size (int): Kernel size for the spatial convolution.
        """
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Define a Conv2D layer to process the concatenated pooling maps.
        self.conv2d = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        # Along the channel axis, compute average and max pooling maps.
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B, M, M, 1)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)     # (B, M, M, 1)
        # Concatenate the two maps: (B, M, M, 2)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        # Apply convolution to get a spatial attention map (B, M, M, 1)
        attn = self.conv2d(concat)
        attn = tf.sigmoid(attn)
        # Multiply the input feature map by the spatial attention map.
        return x * attn

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config


class ECBAM(tf.keras.layers.Layer):
    """
    Efficient Convolutional Block Attention Module (ECBAM).
    Applies efficient channel attention followed by spatial attention.
    """
    def __init__(self, eca_kernel_size=3, sa_kernel_size=7, **kwargs):
        """
        Args:
            eca_kernel_size (int): Kernel size for the efficient channel attention module.
            sa_kernel_size (int): Kernel size for the spatial attention module.
        """
        super(ECBAM, self).__init__(**kwargs)
        self.eca_kernel_size = eca_kernel_size
        self.sa_kernel_size = sa_kernel_size
        self.channel_attention = EfficientChannelAttention(kernel_size=self.eca_kernel_size)
        self.spatial_attention = SpatialAttention(kernel_size=self.sa_kernel_size)

    def call(self, x):
        # First, refine features via channel attention.
        x_out = self.channel_attention(x)
        # Then, apply spatial attention.
        x_out = self.spatial_attention(x_out)
        return x_out

    def get_config(self):
        config = super(ECBAM, self).get_config()
        config.update({
            'eca_kernel_size': self.eca_kernel_size,
            'sa_kernel_size': self.sa_kernel_size,
        })
        return config