import wandb


if __name__ == "__main__":
    run = wandb.init(project="test", name="custom_charts_line")

    x_values = [t for t in range(10)]
    y_values = [x * x for x in x_values]

    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    run.log(
        {"my_custom_plot_id": wandb.plot.line(table, "x", "y", title="Custom Y vs X Line Plot")}
    )
