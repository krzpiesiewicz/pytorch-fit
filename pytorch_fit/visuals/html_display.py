from IPython.display import HTML


def display_network(net, net_name):
    output = f"<p><b>{net_name}:</b></p><table><tr><th>{net}".replace(
        "Sequential(\n", ""
    )
    output = output[:-2].replace("\n", "</td></tr><tr><th>")
    output = output.replace("):", "</th><td style='text-align: left'>")
    output = output + "</td></td></table>"
    output = output.replace("<th>  (", "<th>")
    return HTML(output)


def display_last_history(history):
    output = "<ul>"
    for metric_name, instances in history.items():
        if metric_name != "Confusion":
            output += f"<li>{metric_name}:<ul>"
            for metric_instance, values in instances.items():
                output += f"<li>{metric_instance}: {values[-1]:.3f}</li>"
            output += "</ul></li>"
    output += "</ul>"
    return HTML(output)


def format_vertical_headers(df):
    """Display a dataframe with vertical column headers"""
    styles = [
        dict(selector="th", props=[("width", "40px")]),
        dict(
            selector="th.col_heading",
            props=[
                ("writing-mode", "vertical-rl"),
                ("transform", "rotateZ(180deg)"),
                ("height", "100px"),
                ("vertical-align", "top"),
            ],
        ),
    ]
    return df.fillna("").style.set_table_styles(styles)