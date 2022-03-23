import random
from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        super(CustomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.d_type = ['video_set', 'image_set']
        self.d_len = self.data_source.cummulative_sizes
        self.d_len = [self.d_len[0], self.d_len[1] - self.d_len[0]]
        (self.vs_len, self.is_len) = (self.d_len[0], self.d_len[1]) if self.d_len[0] < self.d_len[1] else (self.d_len[1], self.d_len[0])
        self.indices_dict = {'video_set': list(range(0, self.vs_len)), 'image_set': list(range(self.vs_len, self.is_len + self.vs_len))}

    def __iter__(self):
        self.indices = []
        for idx in range(len(self.d_type)):
            random.shuffle(self.indices_dict[self.d_type[idx]])
        n = len(self.indices_dict[self.d_type[0]]) // self.batch_size
        ratio = len(self.indices_dict[self.d_type[1]]) // (n * self.batch_size)
        for i in range(n):
            self.indices.extend(random.sample(self.indices_dict[self.d_type[0]], self.batch_size))
            for j in range(ratio):
                self.indices.extend(random.sample(self.indices_dict[self.d_type[1]], self.batch_size))

        return iter(self.indices)

    def __len__(self):
        length = 0
        n = self.vs_len // self.batch_size
        length += n * self.batch_size
        ratio = self.is_len // length
        length += length * ratio
        return length
