import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch


def helper(value, j):
    '''
    helper function for data_plot()
    '''
    if value == "None":
        return None
    elif type(value) == list and j < len(value):
        return value[j]
    else:  # not a list so only one value
        if j == 0:
            return value
        else:
            return None


def data_plot(data=None, x=None, y=None,
              x_title=None, y_title=None, title=None,
              **kwargs):
    '''
    list of pandas.DataFrame, list of str, list of str, list of str, kwargs -> plotly plot object

    Precondition: If an argument has multiple objects, they must be in a list (can have nested lists).
                  The order of the arguments must be in the same order as the DataFrames.
                  There must be the same number of x columns as y columns passed.

                  ex) ocv_plot(
                      data = [df1, df2],
                      x = [ "SOC", "SOC-Dis" ],
                      y = [ "OCV", "OCV-Dis" ],
                      mode = ["lines+markers", "markers"],
                      color = ["mintcream", "darkorchid"]
                      )

    This function takes one or more DataFrames, columns from the respective DataFrames to be plot on x and y-axes.
    It also takes the mode of plotting desired for the DataFrames and optional keyword arguments.
    It outputs a plotly plot of the data from the columns that were passed.

    Parameters:
    `data` DataFrame or list of DataFrames

    `x` list of columns or nested lists of columns
        example of each option in order:
            x = ["SOC-Dis"]
            x = ["SOC-Dis","SOC-Chg","SOC"]
            x = [ ["Test Time (sec)","Step Time (sec)"], "Step"]
                Test Time and Step Time are both from the same DataFrame; there must be two y columns as well.

    `y` list of columns or nested lists of columns
        View `x` for help

    `x_title` str
        the name of the x_axis to be displayed
        else None

    `y_title` str
        the name of the y_axis to be displayed
        else None

    `title` str
        The title of the Plot
        default None will not add a title

    **kwargs: (alphabetical order)

    `color` str, list of str, nested lists of str:
        same principle as above arguments,
        assigns the color of the individual data lines.
        if no value is passed for a plot, plotly will do it automatically.

        The 'color' property is a color and may be specified as:
          - A hex string (e.g. '#ff0000')
          - An rgb/rgba string (e.g. 'rgb(255,0,0)')
          - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
          - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
          - A named CSS color:
                aliceblue, antiquewhite, aqua, aquamarine, azure,
                beige, bisque, black, blanchedalmond, blue,
                blueviolet, brown, burlywood, cadetblue,
                chartreuse, chocolate, coral, cornflowerblue,
                cornsilk, crimson, cyan, darkblue, darkcyan,
                darkgoldenrod, darkgray, darkgrey, darkgreen,
                darkkhaki, darkmagenta, darkolivegreen, darkorange,
                darkorchid, darkred, darksalmon, darkseagreen,
                darkslateblue, darkslategray, darkslategrey,
                darkturquoise, darkviolet, deeppink, deepskyblue,
                dimgray, dimgrey, dodgerblue, firebrick,
                floralwhite, forestgreen, fuchsia, gainsboro,
                ghostwhite, gold, goldenrod, gray, grey, green,
                greenyellow, honeydew, hotpink, indianred, indigo,
                ivory, khaki, lavender, lavenderblush, lawngreen,
                lemonchiffon, lightblue, lightcoral, lightcyan,
                lightgoldenrodyellow, lightgray, lightgrey,
                lightgreen, lightpink, lightsalmon, lightseagreen,
                lightskyblue, lightslategray, lightslategrey,
                lightsteelblue, lightyellow, lime, limegreen,
                linen, magenta, maroon, mediumaquamarine,
                mediumblue, mediumorchid, mediumpurple,
                mediumseagreen, mediumslateblue, mediumspringgreen,
                mediumturquoise, mediumvioletred, midnightblue,
                mintcream, mistyrose, moccasin, navajowhite, navy,
                oldlace, olive, olivedrab, orange, orangered,
                orchid, palegoldenrod, palegreen, paleturquoise,
                palevioletred, papayawhip, peachpuff, peru, pink,
                plum, powderblue, purple, red, rosybrown,
                royalblue, rebeccapurple, saddlebrown, salmon,
                sandybrown, seagreen, seashell, sienna, silver,
                skyblue, slateblue, slategray, slategrey, snow,
                springgreen, steelblue, tan, teal, thistle, tomato,
                turquoise, violet, wheat, white, whitesmoke,
                yellow, yellowgreen
          - A number that will be interpreted as a color
            according to scatter.marker.colorscale
          - A list or array of any of the above

    `mode` str, list of str, nested lists of str:
        default None: will set mode = "lines"
        Note: str must be one of "lines", "markers", "lines+markers" which are self-explanatory
        example of each option in order:
            mode = "markers"
            mode = ["lines+markers", "lines"]
            mode = ["lines+markers",["lines","lines"]]

    `name` str, list of str, nested list of strs
        same principle as above arguments
        assigns the names of the individual data lines to be displayed in the legend

    `size` int/float, list of int/float or nested lists of int/float
        same principle as above arguments
        assigns the size of the individual data lines
        if no value is passed, plotly will do it automatically.


    >>>df1 = generate_ocv_pts("JMFM_12_SOC_OCV_Test_220411.txt", to_csv = False)
    >>>df2 = ocv_estimate(df1, to_csv = False)
    >>>data_plot(data = [df1,df2],
          x=[ ["SOC-Chg","SOC-Dis"],"SOC" ],
          y = [ ["OCV-Chg","OCV-Dis"], "OCV" ],
          title = "JMFM-12 OCV vs. SOC Curve",
          x_title = "SOC (%)",
          y_title = "OCV (V)",
          mode = [ ["markers","markers"] ],
          color = [ ["violet","lightcoral"], "darkorchid"],
          name = [ ["Charge-OCV","Discharge-OCV"], "OCV"],
          size = [[4.5,4.5]]
         )
    figure...
    '''
    if type(data) == list and not pd.Series(
        pd.Series([len(x), len(y)]) == len(data)
    ).all():
        return '''Error: x and y columns passed much match the number of DataFrames passed
        Use nested lists for multiple columns from the same DataFrame
        '''

    elif type(data) != list and not pd.Series(pd.Series([len(x), len(y)]) == 1).all():
        return '''Error: x and y columns passed much match the number of DataFrames passed
        Use nested lists for multiple columns from the same DataFrame
        '''

    if "mode" in kwargs.keys():
        if type(kwargs["mode"]) == list and len(kwargs["mode"]) > len(data):
            return "Error: passed more modes than DataFrames"

    if "color" in kwargs.keys():
        if type(kwargs["color"]) == list and len(kwargs["color"]) > len(data):
            return "Error: passed more colors than DataFrames"

    if "name" in kwargs.keys():
        if type(kwargs["name"]) == list and len(kwargs["name"]) > len(data):
            return "Error: passed more names than DataFrames"

    if "size" in kwargs.keys():
        if type(kwargs["size"]) == list and len(kwargs["size"]) > len(data):
            return "Error: passed more sizes than DataFrames"

    frame = pd.DataFrame(data={"x": x, "y": y})

    for i in ["color", "mode", "name", "size"]:
        frame = frame.join(
            pd.Series(kwargs.get(i), name=i, dtype="object"),
            how="outer")

    frame.fillna("None", inplace=True)

    figure = make_subplots(
        x_title=x_title, y_title=y_title, subplot_titles=[title])

    for i in frame.index:
        if type(data) == list:
            use_data = data[i]
        else:
            use_data = data

        if type(frame["x"][i]) == list:  # y[i] must be a list
            for j in range(len(x[i])):
                use_x = frame.loc[i, "x"][j]
                use_y = frame.loc[i, "y"][j]

                use_color = helper(frame.loc[i, "color"], j)
                use_mode = helper(frame.loc[i, "mode"], j)
                use_name = helper(frame.loc[i, "name"], j)
                use_size = helper(frame.loc[i, "size"], j)

                figure.add_trace(
                    go.Scatter(
                        x=use_data[use_x], y=use_data[use_y],
                        mode=use_mode, marker={
                            "size": use_size, "color": use_color},
                        name=use_name)
                )
        else:  # x[i] and y[i] are not lists
            use_x = frame.loc[i, "x"]
            use_y = frame.loc[i, "y"]
            use_color = helper(frame.loc[i, "color"], 0)
            use_mode = helper(frame.loc[i, "mode"], 0)
            use_name = helper(frame.loc[i, "name"], 0)
            use_size = helper(frame.loc[i, "size"], 0)
            # zero is just a placholder

            figure.add_trace(
                go.Scatter(
                    x=use_data[use_x], y=use_data[use_y],
                    mode=use_mode, marker={
                        "size": use_size, "color": use_color},
                    name=use_name)
            )
    return figure


# -------------------------------------------------------

def normalize(data: pd.DataFrame):
    '''
    pd.DataFrame -> pd.DataFrame
    Precondition: "delta t" is removed from the DataFrame

    Normalizes the data by applying sklearn.preprocessing functions
    Voltage is scaled between 0 and 1;
    Current is scaled between -1 and 1;
    SOC is scaled between 0 and 1 (just divided by 100)

    Output:
        normalized pd.DataFrame
    '''
    data["current"] = MaxAbsScaler().fit_transform(
        data["current"].values.reshape(-1, 1))
    data["voltage"] = MinMaxScaler((0, 1)).fit_transform(
        data["voltage"].values.reshape(-1, 1))
    data["soc"] /= 100.

    print(f'''Scaled stats:

variance:\n{data.var(axis = 0)},

mean:\n{data.mean(axis=0)}''')

    return data

# -------------------------------------------------------


def rolling_split_trial(df, window_size):
    '''
    implements rolling window sectioning
    There are four input features: delta_t, V, I at time t, and SOC at time t-1
    Prediction at time t uses the features given
    '''
    if "delta t" in df.columns:
        col = ["delta t", "current", "voltage"]
    else:
        col = ["current", "voltage"]
    df_x = (df[col].iloc[1:].reset_index(drop=True)  # staggered right by one
            .join(
        df["soc"].iloc[:-1].reset_index(drop=True),  # staggered left by one
        how="outer"
    ))
    df_x = [window.values
            for window
            in df_x.rolling(window=window_size,
                            min_periods=window_size - 2,
                            method="table"
                            )][window_size:]

    # staggered right by one
    df_y = df["soc"].iloc[window_size + 1:].values[:, np.newaxis]

    return np.array(df_x, dtype="float32"), np.array(df_y, dtype="float32")


def rolling_split(df, window_size, test_size=0.1, train=True):
    '''
    Precondition: "delta t" is not in the columns
    implements rolling window sectioning
    Four input features: delta_t, I, V, SOC all at time t-1
    The prediction of SOC at time t uses no other information

    Returns a shuffled and windowed dataset using
    sklearn.model_selection.train_test_split

    Parameters:
    `window_size` int
        the number of consecutive data points needed to form a data window
    `test_size` float in between 0 and 0.2 exclusive
        the ratio of data points allocated to the dev/test set
        Should never exceed 0.2
    '''
    assert "delta t" not in df.columns
    assert isinstance(test_size, float)
    assert test_size > 0 and test_size <= 0.2

    df_x = [window.values
            for window
            # staggered left by one
            in df[["current", "voltage", "soc"]].iloc[:-1]
            .rolling(window=window_size,
                     min_periods=window_size - 2,
                     method="table"
                     )][window_size:]

    df_y = df["soc"].iloc[window_size + 1:].values

    if train:
        return train_test_split(np.array(df_x, dtype="float32"),
                                np.array(df_y, dtype="float32")[:, np.newaxis],
                                test_size=test_size,
                                shuffle=True)
    else:
        return (np.array(df_x, dtype="float32"),
                np.array(df_y, dtype="float32")[:, np.newaxis])

# ----------------------------------------------------------------
# Validation


def validate(model, dataloader, dev=True):
    '''
    pytorch model, pytorch DataLoader -> pd.DataFrame, prints 2 tensors and a Plotly plot

    This function runs a dataloader through the model and prints the max and min
    predicted SOC, it also prints a Plotly plot of the predictions versus the labels
    This function outputs a pandas.DataFrame of the predictions with their corresponding labels.
    '''
    pred = []
    with torch.no_grad():
        for x, y in dataloader:
            pred.append(model(x))

    aggregate = []
    for i in pred:  # this way is faster than list comprehension below
        aggregate.extend(i)
    print(max(aggregate), min(aggregate))

    # aggregate = [unit for batch in pred for unit in batch]
    # print(max(aggregate), min(aggregate))

    np_aggregate = np.array([p.detach().cpu().numpy() for p in aggregate])
    np_labels = torch.clone(dataloader.dataset.labels).detach().cpu().numpy()[
        :len(np_aggregate)]

    visualize = pd.DataFrame(data={"pred": np_aggregate.squeeze(),
                                   "labels": np_labels.squeeze()})
    if dev:  # if it is the dev set, the values need to be sorted by value
        visualize.sort_values("labels", inplace=True)
    # if it is the entire dataset, it is already sorted chronologically which is more important

    visualize.reset_index(drop=True)

    visualize["point"] = list(range(1, len(visualize) + 1))

    fig = data_plot(data=visualize,
                    x=[["point", "point"]],
                    y=[["pred", "labels"]],
                    x_title="Data Point",
                    y_title="SOC",
                    title="Predicted vs Actual SOC",
                    name=[["predictions", "labels"]],
                    mode=[["markers", "markers"]],
                    color=[["red", "yellow"]]
                    )
    fig.show()
    return visualize
