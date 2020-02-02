使用tensorBoard可视化词嵌入 embedding时，会涉及到keras版本问题

使用keras时有3种：

1. `import keras`
2. `from tensorflow import keras`	
3. `from tensorflow.python import keras`

尽量统一使用1，但自己编译的`tensorflow-1.13.2-cp35-cp35m-linux_x86_64.whl`上，使用1会报错，该问题是初始化错误，

```
FailedPreconditionError (see above for traceback): Error while reading resource variable Variable from Container: localhost. This could mean that the variable was uninitialized. Not found: Resource localhost/Variable/N10tensorflow3VarE does not exist.
	 [[node embed_embedding/Initializer/ReadVariableOp (defined at /usr/local/lib/python3.5/dist-packages/keras/backend/tensorflow_backend.py:620) ]]
```

于是使用2或3，能顺利学习第一轮，但存储时：

```
AttributeError: 'TensorBoard' object has no attribute 'sess'
```

[原因](https://github.com/tensorflow/tensorboard/issues/1666)

临时解决方案，重写了__init__方法，但还是不行，不知编写更高版本(1.14.0以后)，可否解决该问题

```
class TensorBoardWithSession(tf.keras.callbacks.TensorBoard):

    def __init__(self, **kwargs):
        from tensorflow.python.keras import backend as K
        self.sess = K.get_session()

        super().__init__(**kwargs)

 tf.keras.callbacks.TensorBoard = TensorBoardWithSession
```