


class DNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, hidden_layers=3):
        super(DNN, self).__init()

        layers = []
        layers += [nn.Linear(input_dim, hidden_dim), nn.ELU()]

        for i in range(self.hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]

        layers += [nn.Linear(hidden_dim, output_dim), nn.ELU()]

        self.network = nn.Sequential(*layers)

    def forward(self, data):
        x = data.x

        return self.network(x).squeeze(-1)
