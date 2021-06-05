from torch.utils.data import ConcatDataset, Dataset
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import pdb


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, args, original_dataset, sub_labels, class_per_task=None, task_id=None, target_transform=None, all_labels=None, cifar=False):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        self.args = args
        self.task_id = task_id
        self.class_per_task = class_per_task

        if cifar:
            ## load data faster
            self.sub_indeces = [index for index, label in enumerate(all_labels) if label in sub_labels]
        else:
            for index in range(len(self.dataset)):
                if hasattr(original_dataset, "train_labels"):
                    if self.dataset.target_transform is None:
                        label = self.dataset.train_labels[index]
                    else:
                        label = self.dataset.target_transform(self.dataset.train_labels[index])
                elif hasattr(self.dataset, "test_labels"):
                    if self.dataset.target_transform is None:
                        label = self.dataset.test_labels[index]
                    else:
                        label = self.dataset.target_transform(self.dataset.test_labels[index])
                else:
                    label = self.dataset[index][1]
                if label in sub_labels:
                    self.sub_indeces.append(index)

        self.target_transform = target_transform

        
        

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)

        if self.args.random_task_order:
            new_label = (sample[1]%self.class_per_task) + self.task_id*self.class_per_task
            return (sample[0], new_label)
        else:
            return sample



def _get_linear_line(start, end, direction="up"):
    if direction == "up":
        return torch.FloatTensor([(i - start)/(end-start) for i in range(start, end)])
    return torch.FloatTensor([1 - ((i - start) / (end - start)) for i in range(start, end)])

def _create_task_probs(iters, tasks, task_id, beta=3):
    if beta <= 1:
        peak_start = int((task_id/tasks)*iters)
        peak_end = int(((task_id + 1) / tasks)*iters)
        start = peak_start
        end = peak_end
    else:
        start = max(int(((beta*task_id - 1)*iters)/(beta*tasks)), 0)
        peak_start = int(((beta*task_id + 1)*iters)/(beta*tasks))
        peak_end = int(((beta * task_id + (beta - 1)) * iters) / (beta * tasks))
        end = min(int(((beta * task_id + (beta + 1)) * iters) / (beta * tasks)), iters)

    probs = torch.zeros(iters, dtype=torch.float)
    if task_id == 0:
        probs[start:peak_start].add_(1)
    else:
        probs[start:peak_start] = _get_linear_line(start, peak_start, direction="up")
    probs[peak_start:peak_end].add_(1)
    if task_id == tasks - 1:
        probs[peak_end:end].add_(1)
    else:
        probs[peak_end:end] = _get_linear_line(peak_end, end, direction="down")
    return probs



class Sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, tasks_samples_indices, tasks_probs_over_iterations, samples_in_batch):
        self.data_source = data_source
        self.tasks_samples_indices = tasks_samples_indices
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.samples_in_batch = samples_in_batch
        self.iter_num = 0
        pdb.set_trace()
    
    def __iter__(self):
        tsks = Categorical(probs=self.tasks_probs_over_iterations[self.iter_num]).sample(torch.Size([self.samples_in_batch]))

        self.generate_iters_indices(self.num_of_batches)
        self.current_iteration += self.num_of_batches
        return iter([item for sublist in self.iter_indices_per_iteration[self.current_iteration - self.num_of_batches:self.current_iteration] for item in sublist])

    def __len__(self):
        return len(self.samples_in_batch)


class ContinuousMultinomialSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    self.tasks_probs_over_iterations is the probabilities of tasks over iterations.
    self.samples_distribution_over_time is the actual distribution of samples over iterations
                                            (the result of sampling from self.tasks_probs_over_iterations).
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, samples_in_batch=128, num_of_batches=69, tasks_samples_indices=None,
                 tasks_probs_over_iterations=None):


        self.data_source = data_source
        assert tasks_samples_indices is not None, "Must provide tasks_samples_indices - a list of tensors," \
                                                  "each item in the list corrosponds to a task, each item of the " \
                                                  "tensor corrosponds to index of sample of this task"
        self.tasks_samples_indices = tasks_samples_indices
        self.num_of_tasks = len(self.tasks_samples_indices)
        assert tasks_probs_over_iterations is not None, "Must provide tasks_probs_over_iterations - a list of " \
                                                         "probs per iteration"
        assert all([isinstance(probs, torch.Tensor) and len(probs) == self.num_of_tasks for
                    probs in tasks_probs_over_iterations]), "All probs must be tensors of len" \
                                                              + str(self.num_of_tasks) + ", first tensor type is " \
                                                              + str(type(tasks_probs_over_iterations[0])) + ", and " \
                                                              " len is " + str(len(tasks_probs_over_iterations[0]))
        self.tasks_probs_over_iterations = tasks_probs_over_iterations
        self.current_iteration = 0

        self.samples_in_batch = samples_in_batch
        self.num_of_batches = num_of_batches

        # Create the samples_distribution_over_time
        self.samples_distribution_over_time = [[] for _ in range(self.num_of_tasks)]
        self.iter_indices_per_iteration = []

        if not isinstance(self.samples_in_batch, int) or self.samples_in_batch <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.samples_in_batch))

    def generate_iters_indices(self, num_of_iters):
        from_iter = len(self.iter_indices_per_iteration)
        for iter_num in range(from_iter, from_iter+num_of_iters):

            # Get random number of samples per task (according to iteration distribution)
            tsks = Categorical(probs=self.tasks_probs_over_iterations[iter_num]).sample(torch.Size([self.samples_in_batch]))
            # Generate samples indices for iter_num
            iter_indices = torch.zeros(0, dtype=torch.int32)
            for task_idx in range(self.num_of_tasks):
                if self.tasks_probs_over_iterations[iter_num][task_idx] > 0:
                    num_samples_from_task = (tsks == task_idx).sum().item()
                    self.samples_distribution_over_time[task_idx].append(num_samples_from_task)
                    # Randomize indices for each task (to allow creation of random task batch)
                    tasks_inner_permute = np.random.permutation(len(self.tasks_samples_indices[task_idx]))
                    rand_indices_of_task = tasks_inner_permute[:num_samples_from_task]
                    iter_indices = torch.cat([iter_indices, self.tasks_samples_indices[task_idx][rand_indices_of_task]])
                else:
                    self.samples_distribution_over_time[task_idx].append(0)
            self.iter_indices_per_iteration.append(iter_indices.tolist())

    def __iter__(self):
        
        self.generate_iters_indices(self.num_of_batches)
        self.current_iteration += self.num_of_batches
        return iter([item for sublist in self.iter_indices_per_iteration[self.current_iteration - self.num_of_batches:self.current_iteration] for item in sublist])

    def __len__(self):
        return len(self.samples_in_batch)

