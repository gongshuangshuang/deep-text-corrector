from tensorflow.python.framework import dtypes, ops
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import control_flow_ops, embedding_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python import shape
from tensorflow.python.ops import init_ops
# 获取一个循环函数，该函数提取并嵌入前一个符号
def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """
    :param embedding:embedding tensor for symbols.
    :param output_projection:None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
    :param update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.
    :return: A loop function.
    """
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev, prev_symbol
    return loop_function
# 线性函数
def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

        # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
        else:
            total_arg_size += shape[1]
    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with variable_scope.variable_scope(scope or "Linear"):
        matrix = variable_scope.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            # array_ops.concat(1,args),将inputs和state按列连接起来，其实是增加了inputs的特征维度，将前一个状态中的信息放在当前状态中，也就增加了inputs的信息量，比如inputs=[[1,2,3],[4,5,6]],state=[[7,8,9,10],[11,12,13,14]], array_ops.concat(1,[inputs,state])=[[1,2,3,7,8,9,10],[4,5,6,11,12,13,14]],输入的特征维度从3增加到了7
            # matmul(x,w)
            res = math_ops.matmul(array_ops.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = variable_scope.get_variable("Bias", [output_size], dtype=dtype,
                                    initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    # matmul(x,w)+b
    return res + bias_term

# ---------------------------------------编码器-----------------------------------------

#----版本1----
# 最简单版本，输入和输出都是embedding的形式；最后一步的state vector作为decoder的initial
# state；encoder和decoder用相同的RNN cell， 但不共享权值参数；
def basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, scope=None):
    """
    :param encoder_inputs:A list of 2D Tensors [batch_size x input_size].
    :param decoder_inputs:A list of 2D Tensors [batch_size x input_size].
    :param cell:rnn_cell.RNNCell defining the cell function and size.
    :param dtype:The dtype of the initial state of the RNN cell (default: tf.float32).
    :param scope:VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
    :return:A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell in the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
        _, enc_state = rnn.static_rnn(cell, encoder_inputs, dtype=dtype)
        return rnn_decoder(decoder_inputs, enc_state, cell)

#----版本2----
# 同basic_rnn_seq2seq，但encoder和decoder共享权值参数
def tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, loop_function=None,
                     dtype=dtypes.float32, scope=None):
    with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
        scope = scope or "tied_rnn_seq2seq"
        _, enc_state = rnn.static_rnn(cell, encoder_inputs, dtype=dtype, scope=scope)
        variable_scope.variable_scope.get_variable_scope().reuse_variables()
        return rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loop_function, scope=scope)

#----版本3----
# 同basic_rnn_seq2seq，但输入和输出改为id的形式，函数会在内部创建分别用于encoder和decoder的embedding matrix
def embedding_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None):
    with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype
        # Encoder
        encoder_cell = rnn.EmbeddingWrapper(cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)
        # Decoder
        if output_projection is None:
            cell = rnn.OutputProjectionWrapper(cell, num_decoder_symbols)
        if isinstance(feed_previous, bool):
            return embedding_rnn_decoder(decoder_inputs,
                encoder_state,
                cell,
                num_decoder_symbols,
                embedding_size,
                output_projection=output_projection,
                feed_previous=feed_previous)
        # 如果feed_previous是张量，我们构造2个图并进行cond
        def decoder(feed_previous_bool):
            if feed_previous_bool:
                reuse = None
            else:
                reuse = True
            with variable_scope.variable_scope(variable_scope.variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = embedding_rnn_decoder(decoder_inputs, encoder_state, cell, num_decoder_symbols,
                    embedding_size, output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list
        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)    # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state, flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state

#----版本4----
# 同tied_rnn_seq2seq，但输入和输出改为id的形式，函数会在内部创建分别用于encoder和decoder的embedding matrix
def embedding_tied_rnn_seq2seq(encoder_inputs,
                               decoder_inputs,
                               cell,
                               num_symbols,
                               embedding_size,
                               num_decoder_symbols=None,
                               output_projection=None,
                               feed_previous=False,
                               dtype=None,
                               scope=None):
    with variable_scope.variable_scope(scope or "embedding_tied_rnn_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
        proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])
        embedding = variable_scope.variable_scope.get_variable("embedding", [num_symbols, embedding_size], dtype=dtype)
        emb_encoder_inputs = [embedding_ops.embedding_lookup(embedding, x)
                              for x in encoder_inputs]
        emb_decoder_inputs = [embedding_ops.embedding_lookup(embedding, x)
                              for x in decoder_inputs]
        output_symbols = num_symbols
        if num_decoder_symbols is not None:
            output_symbols = num_decoder_symbols
        if output_projection is None:
            cell = rnn.OutputProjectionWrapper(cell, output_symbols)
        if isinstance(feed_previous, bool):
            loop_function = _extract_argmax_and_embed(
                embedding, output_projection, True) if feed_previous else None
            return tied_rnn_seq2seq(emb_encoder_inputs, emb_decoder_inputs, cell,
                                    loop_function=loop_function, dtype=dtype)
        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            loop_function = _extract_argmax_and_embed(
                embedding, output_projection, False) if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.variable_scope.get_variable_scope(),reuse=reuse):
                outputs, state = tied_rnn_seq2seq(
                    emb_encoder_inputs, emb_decoder_inputs, cell,
                    loop_function=loop_function, dtype=dtype)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list
        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)   # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        # Calculate zero-state to know it's structure.
        static_batch_size = encoder_inputs[0].get_shape()[0]
        for inp in encoder_inputs[1:]:
            static_batch_size.merge_with(inp.get_shape()[0])
        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = array_ops.shape(encoder_inputs[0])[0]
        zero_state = cell.zero_state(batch_size, dtype)
        if nest.is_sequence(zero_state):
            state = nest.pack_sequence_as(structure=zero_state,
                                          flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state

#----版本5----
# 同embedding_rnn_seq2seq，但多了attention机制
# T 代表time_steps, 时序长度
def embedding_attention_seq2seq(encoder_inputs,  # [T, batch_size]
                                decoder_inputs,  # [T, batch_size]
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,     # attention的抽头数量
                                output_projection=None,  #decoder的投影矩阵
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                loop_fn_factory=_extract_argmax_and_embed):
    """
    :param encoder_inputs: encoder的输入，int32型 id tensor list
    :param decoder_inputs: decoder的输入，int32型id tensor list
    :param cell: RNN_Cell的实例
    :param num_encoder_symbols: 编码的符号数，即词表大小
    :param num_decoder_symbols: 解码的符号数，即词表大小
    :param embedding_size: 词向量的维度
    :param num_heads: attention的抽头数量，一个抽头算一种加权求和方式
    :param output_projection: decoder的output向量投影到词表空间时，用到的投影矩阵和偏置项(W, B)；W的shape是[output_size, num_decoder_symbols]，B的shape是[num_decoder_symbols]；若此参数存在且feed_previous=True，上一个decoder的输出先乘W再加上B作为下一个decoder的输入
    :param feed_previous: 若为True, 只有第一个decoder的输入（“GO"符号）有用，所有的decoder输入都依赖于上一步的输出；一般在测试时用
    :param dtype:
    :param scope:
    :param initial_state_attention: 默认为False, 初始的attention是零；若为True，将从initial state和attention states开始attention
    :param loop_fn_factory:
    :return:
    """
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        # 创建了一个embedding matrix.
        # 计算encoder的output和state
        # 生成attention states，用于计算attention
        encoder_cell = rnn.EmbeddingWrapper(  # EmbeddingWrapper, 是RNNCell的前面加一层embedding，作为encoder_cell, input就可以是word的id.
            cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.static_rnn(
            encoder_cell, encoder_inputs, dtype=dtype)  #  [T，batch_size，size]

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]  # T * [batch_size, 1, size]
        attention_states = array_ops.concat(top_states, 1)  # [batch_size,T,size]

        # Decoder.
        # 生成decoder的cell，通过OutputProjectionWrapper类对输入参数中的cell实例包装实现
        output_size = None
        if output_projection is None:
            cell = rnn.OutputProjectionWrapper(cell, num_decoder_symbols)  # OutputProjectionWrapper将输出映射成想要的维度
            output_size = num_decoder_symbols
        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                decoder_inputs,
                encoder_state,
                attention_states,
                cell,
                num_decoder_symbols,
                embedding_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention,
                loop_fn_factory=loop_fn_factory)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=initial_state_attention,
                    loop_fn_factory=loop_fn_factory)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state,
                                          flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state

#----版本6----
# One-to-many RNN sequence-to-sequence model (multi-task)
def one2many_rnn_seq2seq(encoder_inputs,
                         decoder_inputs_dict,
                         cell,
                         num_encoder_symbols,
                         num_decoder_symbols_dict,
                         embedding_size,
                         feed_previous=False,
                         dtype=None,
                         scope=None):
    outputs_dict = {}
    state_dict = {}

    with variable_scope.variable_scope(
            scope or "one2many_rnn_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype

        # Encoder.
        encoder_cell = rnn.EmbeddingWrapper(
            cell, embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # Decoder.
        for name, decoder_inputs in decoder_inputs_dict.items():
            num_decoder_symbols = num_decoder_symbols_dict[name]

            with variable_scope.variable_scope("one2many_decoder_" + str(name)) as scope:
                decoder_cell = rnn.OutputProjectionWrapper(cell, num_decoder_symbols)
                if isinstance(feed_previous, bool):
                    outputs, state = embedding_rnn_decoder(
                        decoder_inputs, encoder_state, decoder_cell, num_decoder_symbols,
                        embedding_size, feed_previous=feed_previous)
                else:
                    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
                    def filled_embedding_rnn_decoder(feed_previous):
                        """The current decoder with a fixed feed_previous parameter."""
                        # pylint: disable=cell-var-from-loop
                        reuse = None if feed_previous else True
                        vs = variable_scope.get_variable_scope()
                        with variable_scope.variable_scope(vs, reuse=reuse):
                            outputs, state = embedding_rnn_decoder(
                                decoder_inputs, encoder_state, decoder_cell,
                                num_decoder_symbols, embedding_size,
                                feed_previous=feed_previous)
                        # pylint: enable=cell-var-from-loop
                        state_list = [state]
                        if nest.is_sequence(state):
                            state_list = nest.flatten(state)
                        return outputs + state_list

                    outputs_and_state = control_flow_ops.cond(
                        feed_previous,
                        lambda: filled_embedding_rnn_decoder(True),
                        lambda: filled_embedding_rnn_decoder(False))
                    # Outputs length is the same as for decoder inputs.
                    outputs_len = len(decoder_inputs)
                    outputs = outputs_and_state[:outputs_len]
                    state_list = outputs_and_state[outputs_len:]
                    state = state_list[0]
                    if nest.is_sequence(encoder_state):
                        state = nest.pack_sequence_as(structure=encoder_state,
                                                      flat_sequence=state_list)
            outputs_dict[name] = outputs
            state_dict[name] = state

    return outputs_dict, state_dict

# ---------------------------------------解码器-----------------------------------------

#----版本1----
def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
    """
    :param decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    :param initial_state:2D Tensor with shape [batch_size x cell.state_size].
    :param cell:rnn_cell.RNNCell defining the cell function and size.
    :param loop_function:If not None, this function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
    :param scope:VariableScope for the created subgraph; defaults to "rnn_decoder".
    :return:The state of each cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
          (Note that in some cases, like basic RNN cell or GRU cell, outputs and
           states can be the same. They are different for LSTM cells though.)
    """
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state

#----版本2----
# 带嵌入向量和纯解码选项的RNN解码器
def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
    with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
        if output_projection is not None:
            dtype = scope.dtype
            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])
        embedding = variable_scope.variable_scope.get_variable("embedding", [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = (embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)
        return rnn_decoder(emb_inp, initial_state, cell,
                           loop_function=loop_function)

#----版本3----
# 带嵌入向量、注意力机制和纯解码选项的RNN解码器
# 第一步创建了解码用的embedding；
# 第二步创建了一个循环函数loop_function，用于将上一步的输出映射到词表空间，输出一个word embedding作为下一步的输入；
# 最后attention_decoder完成解码工作！
def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False,
                                loop_fn_factory=_extract_argmax_and_embed):
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])
    with variable_scope.variable_scope(scope or "embedding_attention_decoder", dtype=dtype) as scope:
        embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
        loop_function = loop_fn_factory(embedding, output_projection,
                                        update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(
            emb_inp,
            initial_state,
            attention_states,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)

#----版本4----
# 带有注意力机制的RNN编码器
# attention，就是在每个解码的时间步，对encoder的隐层状态进行加权求和，针对不同信息进行不同程度的注意力.
# 分为三步：（1）通过当前隐层状态(d_{t})和关注的隐层状态(h_{i})求出对应权重u^{t}_{i}；
# （2）softmax归一化为概率；
# （3）作为加权系数对不同隐层状态求和，得到一个的信息向量d^{'}_{t}。后续的d^{'}_{t}使用会因为具体任务有所差别.
def attention_decoder(decoder_inputs,  # T * [batch_size, input_size]
                      initial_state,  # [batch_size, cell.states]
                      attention_states,  # [batch_size, attn_length , attn_size]
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        # W_{1}*h_{i}用的是卷积的方式实现，返回的tensor的形状是[batch_size, attn_length, 1, attention_vec_size]
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state
        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            # W_{2}*d_{t}，此项是通过下面的线性映射函数linear实现
            for a in range(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    # query对应当前隐层状态d_t
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    # 计算u_t
                    s = math_ops.reduce_sum(
                        v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                        [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in range(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp, inp_symbol = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

        return outputs, state

# ---------------------------------------损失函数-----------------------------------------
# 计算所有examples的加权交叉熵损失，logits参数是一个2D Tensor构成的列表对象，
# 每一个2D Tensor的尺寸为[batch_size x num_decoder_symbols]，函数的返回值是一个1D float类型的Tensor，尺寸为batch_size，
# 其中的每一个元素代表当前输入序列example的交叉熵
def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                        logits + targets + weights):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    logit, target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps

# sequence_loss对sequence_loss_by_example函数返回的结果进行了一个tf.reduce_sum运算，因此返回的是一个标称型float Tensor
def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    with ops.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, cost.dtype)
        else:
            return cost
# ---------------------------------------bucket模型-----------------------------------------
# 创建了一个支持bucketing策略的sequence-to-sequence模型，它仍然属于Graph的定义阶段。
# 具体来说，这段程序定义了length(buckets)个graph，每个graph的输入为总模型的输入“占位符”的一部分，
# 但这些graphs共享模型参数，函数的返回值outputs和losses均为列表对象，尺寸为[length(buckets)]，
# 其中每一个元素为当前graph的bucket_outputs和bucket_loss。
# Bucketing是一种有效处理不同长度的句子的方法
# 使用多个buckets 并且将每个句子填充为对应的bucket的长度
def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, seq2seq, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))
    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):
                bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                            decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                if per_example_loss:
                    losses.append(sequence_loss_by_example(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(sequence_loss(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))
    return outputs, losses