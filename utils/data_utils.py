import time

from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class DALIClassificationIterator(DALIGenericIterator):
    """
    DALI iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensor.

    Calling

    .. code-block:: python

       DALIClassificationIterator(pipelines, reader_name)

    is equivalent to calling

    .. code-block:: python

       DALIGenericIterator(pipelines, ["data", "label"], reader_name)

    Parameters
    ----------
    pipelines : list of nvidia.dali.Pipeline
                List of pipelines to use
    size : int, default = -1
                Number of samples in the shard for the wrapped pipeline (if there is more than
                one it is a sum)
                Providing -1 means that the iterator will work until StopIteration is raised
                from the inside of iter_setup(). The options `last_batch_policy` and
                `last_batch_padded` don't work in such case. It works with only one pipeline inside
                the iterator.
                Mutually exclusive with `reader_name` argument
    reader_name : str, default = None
                Name of the reader which will be queried to the shard size, number of shards and
                all other properties necessary to count properly the number of relevant and padded
                samples that iterator needs to deal with. It automatically sets `last_batch_policy`
                to PARTIAL when the FILL is used, and `last_batch_padded` accordingly to match
                the reader's configuration
    auto_reset : string or bool, optional, default = False
                Whether the iterator resets itself for the next epoch or it requires reset() to be
                called explicitly.

                It can be one of the following values:

                * ``"no"``, ``False`` or ``None`` - at the end of epoch StopIteration is raised
                  and reset() needs to be called
                * ``"yes"`` or ``"True"``- at the end of epoch StopIteration is raised but reset()
                  is called internally automatically

    dynamic_shape : any, optional,
                Parameter used only for backward compatibility.
    fill_last_batch : bool, optional, default = None
                **Deprecated** Please use ``last_batch_policy`` instead

                Whether to fill the last batch with data up to 'self.batch_size'.
                The iterator would return the first integer multiple
                of self._num_gpus * self.batch_size entries which exceeds 'size'.
                Setting this flag to False will cause the iterator to return
                exactly 'size' entries.
    last_batch_policy: optional, default = LastBatchPolicy.FILL
                What to do with the last batch when there are not enough samples in the epoch
                to fully fill it. See :meth:`nvidia.dali.plugin.base_iterator.LastBatchPolicy`
    last_batch_padded : bool, optional, default = False
                Whether the last batch provided by DALI is padded with the last sample
                or it just wraps up. In the conjunction with ``last_batch_policy`` it tells
                if the iterator returning last batch with data only partially filled with
                data from the current epoch is dropping padding samples or samples from
                the next epoch. If set to ``False`` next
                epoch will end sooner as data from it was consumed but dropped. If set to
                True next epoch would be the same length as the first one. For this to happen,
                the option `pad_last_batch` in the reader needs to be set to True as well.
                It is overwritten when `reader_name` argument is provided
    prepare_first_batch : bool, optional, default = True
                Whether DALI should buffer the first batch right after the creation of the iterator,
                so one batch is already prepared when the iterator is prompted for the data

    Example
    -------
    With the data set ``[1,2,3,4,5,6,7]`` and the batch size 2:

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = True  -> last batch = ``[7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded = False -> last batch = ``[7]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = True   -> last batch = ``[7, 7]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.FILL, last_batch_padded = False  -> last batch = ``[7, 1]``,
    next iteration will return ``[2, 3]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = True   -> last batch = ``[5, 6]``,
    next iteration will return ``[1, 2]``

    last_batch_policy = LastBatchPolicy.DROP, last_batch_padded = False  -> last batch = ``[5, 6]``,
    next iteration will return ``[2, 3]``
    """

    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True,
                 label=["data", "label"]):
        super(DALIClassificationIterator, self).__init__(pipelines, label, size,
                                                         reader_name=reader_name,
                                                         auto_reset=auto_reset,
                                                         fill_last_batch=fill_last_batch,
                                                         dynamic_shape=dynamic_shape,
                                                         last_batch_padded=last_batch_padded,
                                                         last_batch_policy=last_batch_policy,
                                                         prepare_first_batch=prepare_first_batch)


