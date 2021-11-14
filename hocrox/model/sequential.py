from prettytable import PrettyTable


class Sequential:
    def __init__(self):
        self.__frozen = False
        self.__layers = []

    def add(self, layer):
        if self.__frozen:
            raise ValueError("Model is frozen")

        self.__layers.append(layer)

    def summary(self):
        t = PrettyTable(["Index", "Name", "Parameters"])

        for index, layer in enumerate(self.__layers):
            (
                name,
                parameters,
            ) = layer.get_description()

            t.add_row(
                [
                    f"#{index+1}",
                    name,
                    parameters,
                ]
            )

        return str(t)

    def transform(self):
        pass

    def transform_generator(self):
        pass

    def freeze(self):
        self.__frozen = True

    def save(self):
        pass

    def load(self):
        pass
