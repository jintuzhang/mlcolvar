import mlcolvar.graph.data.datamodule


"""
The data module for lightning.
"""

__all__ = ['PairDataModule', 'PairCombinedDataModule']


# NOTE: since the PairDataSet is also built upon torch_geometric.data.Data, and
# has nothing to do with the dataset structure, we reuse the `GraphDataModule`
# and `GraphCombinedDataModule` here.


class PairDataModule(mlcolvar.graph.data.GraphDataModule):

    def __repr__(self) -> str:
        result = ''
        n_digits = len(str(self._n_total))
        data_string_1 = '[ \033[32m{{:{:d}d}}\033[0m\033[36m 󰡷 \033[0m'
        data_string_2 = '| \033[32m{{:{:d}d}}\033[0m\033[36m  \033[0m'
        shuffle_string_1 = '|\033[36m  \033[0m ]'
        shuffle_string_2 = '|\033[36m  \033[0m ]'

        prefix = '\033[1m\033[34m  BASEDATA  \033[0m: '
        result += (
            prefix + self._dataset.__repr__().split('PAIRDATASET ')[1] + '\n'
        )
        prefix = '\033[1m\033[34m  TRAINING  \033[0m: '
        string = prefix + data_string_1.format(n_digits)
        result += string.format(
            self._n_train, self._n_train / self._n_total * 100
        )
        string = data_string_2.format(n_digits)
        result += string.format(self.batch_size[0])
        if self.shuffle[0]:
            result += shuffle_string_1
        else:
            result += shuffle_string_2

        if self._n_validation > 0:
            result += '\n'
            prefix = '\033[1m\033[34m VALIDATION \033[0m: '
            string = prefix + data_string_1.format(n_digits)
            result += string.format(
                self._n_validation, self._n_validation / self._n_total * 100
            )
            string = data_string_2.format(n_digits)
            result += string.format(self.batch_size[1])
            if self.shuffle[1]:
                result += shuffle_string_1
            else:
                result += shuffle_string_2

        if self._n_test > 0:
            result += '\n'
            prefix = '\033[1m\033[34m    TEST    \033[0m: '
            string = prefix + data_string_1.format(n_digits)
            result += string.format(
                self._n_test, self._n_test / self._n_total * 100
            )
            string = data_string_2.format(n_digits)
            result += string.format(self.batch_size[2])
            if self.shuffle[2]:
                result += shuffle_string_1
            else:
                result += shuffle_string_2
        return result


class PairCombinedDataModule(mlcolvar.graph.data.GraphCombinedDataModule):

    def __repr__(self) -> str:
        result = ''
        n_digits = len(str(self._n_total))
        data_string_1 = '[ \033[32m{{:{:d}d}}\033[0m\033[36m 󰡷 \033[0m'
        data_string_2 = '| \033[32m{{:{:d}d}}\033[0m\033[36m  \033[0m'
        shuffle_string = '|\033[36m  \033[0m ]'

        prefix = '\033[1m\033[34m  BASEDATA  \033[0m: '
        result += (
            prefix + self._datasets[0].__repr__().split('PAIRDATASET ')[1]
            + '\n'
            + prefix + self._datasets[1].__repr__().split('PAIRDATASET ')[1]
            + '\n'
        )
        prefix = '\033[1m\033[34m  TRAINING  \033[0m: '
        string = prefix + data_string_1.format(n_digits)
        result += string.format(
            self._n_train * 2, self._n_train / self._n_total * 100
        )
        string = data_string_2.format(n_digits)
        result += string.format(self.batch_size[0])
        result += shuffle_string

        if self._n_validation > 0:
            result += '\n'
            prefix = '\033[1m\033[34m VALIDATION \033[0m: '
            string = prefix + data_string_1.format(n_digits)
            result += string.format(
                self._n_validation * 2,
                self._n_validation / self._n_total * 100
            )
            string = data_string_2.format(n_digits)
            result += string.format(self.batch_size[1])
            result += shuffle_string

        if self._n_test > 0:
            result += '\n'
            prefix = '\033[1m\033[34m    TEST    \033[0m: '
            string = prefix + data_string_1.format(n_digits)
            result += string.format(
                self._n_test * 2, self._n_test / self._n_total * 100
            )
            string = data_string_2.format(n_digits)
            result += string.format(self.batch_size[2])
            result += shuffle_string
        return result
