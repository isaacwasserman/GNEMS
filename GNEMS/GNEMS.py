import torch
class Segmentor:
    def __init__(self, d=16, n_filters=16,
                 dropout=0.2, lambda_=0.3, size=(512, 512),
                 lr=0.001, iterations=100, subset_size=0.5,
                 prediction_stride=1, slic_segments=100, sigma=3,
                 seed=0, deterministic=True):
        self.d = d
        self.n_filters = n_filters
        self.dropout = dropout
        self.lambda_ = lambda_
        self.size = size
        self.tile_size = (size[0] // d, size[1] // d)
        self.lr = lr
        self.iterations = iterations
        self.subset_size = subset_size
        self.prediction_stride = prediction_stride
        self.seed = seed
        self.net = None
        self.losses = []
        self.intermediate_partitions = []
        self.intermediate_probabilities = []
        self.intermediate_graphs = []
        self.slic_segments = slic_segments
        self.sigma = sigma
        self.deterministic = deterministic

    def fit(self, image):
        pass
        # self.losses = []
        # self.intermediate_partitions = []
        # self.intermediate_probabilities = []
        # self.intermediate_graphs = []
        # self.image = image
        
        # if self.seed is not None:
        #     torch.manual_seed(self.seed)
        # X = tile(image, d=self.d).type(torch.float32).to(device).permute(0, 3, 1, 2)
        # if self.deterministic:
        #     y_initial = initial_labels(self.d**2).type(torch.float32).to(device).unsqueeze(1)
        # else:
        #     y_initial = arbitrary_labels(self.d**2).type(torch.float32).to(device).unsqueeze(1)

        # self.net = CNN(image_size=self.tile_size, n_filters=self.n_filters, n_channels=X.shape[1], dropout=self.dropout).to(device)
        
        # if self.deterministic:
        #     torch.manual_seed(0)
        #     init = torch.nn.init
        #     init_weights = lambda m: init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu')) if type(m) == nn.Conv2d else None
        #     self.net.apply(init_weights)
        
        # # train CNN
        # y_intermediate = y_initial.clone().detach()
        # criterion = nn.BCELoss()
        # optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # itertimes = []
        # graphtimes = []
        # for iteration in range(self.iterations):
        #     start = time()
        #     # extract subset of data
        #     if not self.deterministic:
        #         shuffled_idx = torch.randperm(X.shape[0])
        #     else:
        #         shuffled_idx = torch.arange(X.shape[0])
        #     X_shuffled = X[shuffled_idx]
        #     y_intermediate_shuffled = y_intermediate[shuffled_idx]
        #     if not self.deterministic:
        #         X_subset = X_shuffled[:int(self.subset_size * X.shape[0])]
        #         y_intermediate_subset = y_intermediate_shuffled[:int(self.subset_size * X.shape[0])]
        #     else:
        #         X_subset = X_shuffled[iteration % int(1 / self.subset_size)::int(1 / self.subset_size)]
        #         y_intermediate_subset = y_intermediate_shuffled[iteration % int(1 / self.subset_size)::int(1 / self.subset_size)]

        #     inputs = X_subset
        #     labels = y_intermediate_subset
        #     optimizer.zero_grad()
        #     outputs = self.net(inputs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()

        #     # update y_intermediate
        #     probabilities = self.net(X).detach().squeeze(1).cpu().numpy()
        #     graph_start = time()
        #     partition, g = graph_cut(self.d, probabilities, self.lambda_)
        #     graphtimes.append(time() - graph_start)
        #     y_intermediate = torch.Tensor(partition).type(torch.float32).to(device).unsqueeze(1)

        #     # save intermediate results
        #     self.losses.append(loss.item() / X.shape[0])
        #     self.intermediate_partitions.append(partition)
        #     self.intermediate_probabilities.append(probabilities)
        #     self.intermediate_graphs.append(g)
        #     itertimes.append(time() - start)
        # print("Average training time per iteration:", np.array(itertimes).mean())
        # print("Average graph cut time:", np.array(graphtimes).mean())

    def predict(self, show_progress=True):
        pass
        # stride = self.prediction_stride
        # image = self.image
        # image_tensor = torch.tensor(image, dtype=torch.float32)
        # all_tiles = image_tensor.unfold(0, self.tile_size[0], stride).unfold(1, self.tile_size[1], stride).reshape(-1, 1, self.tile_size[0], self.tile_size[1]).to(device)

        # all_tiles_ds = TileDS(all_tiles)

        # # set up dataloaders
        # batch_size = 4096
        # if device == "cuda":
        #     batch_size = 16384
        # loader = torch.utils.data.DataLoader(all_tiles_ds, batch_size=batch_size, shuffle=False)
        # n_batches = len(loader)

        # with torch.no_grad():
        #     all_tiles_predictions = torch.zeros((len(all_tiles_ds))).to(device)
        #     iterator = tqdm(enumerate(loader), total=n_batches) if show_progress else enumerate(loader)
        #     for batch_i, batch in iterator:
        #         batch_predictions = self.net(batch)
        #         all_tiles_predictions[batch_i*loader.batch_size:(batch_i+1)*loader.batch_size] = batch_predictions.squeeze(1)

        # predictions = all_tiles_predictions.reshape(((self.size[0] - self.tile_size[0]) // stride) + 1, ((self.size[1] - self.tile_size[1]) // stride) + 1)
        # pixelwise_probabilities = torch.nn.functional.interpolate(predictions.unsqueeze(0).unsqueeze(0), size=image.shape, mode='bilinear', align_corners=True)
        # pixelwise_probabilities -= pixelwise_probabilities.min()
        # pixelwise_probabilities /= pixelwise_probabilities.max()
        # pixelwise_probabilities *= 255
        # pixelwise_probabilities = pixelwise_probabilities.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        # # return pixelwise_probabilities
        # grayscale = False
        # if 3 not in image.shape:
        #     grayscale = True
        #     image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # segments = slic(image, n_segments=self.slic_segments, sigma=self.sigma)
        # segmentation = color.label2rgb(segments, pixelwise_probabilities, kind='avg', bg_label=0)
        # segmentation = segmentation > auto_threshold(segmentation)
        # if grayscale:
        #     segmentation = segmentation[:, :, 0]
        # return segmentation