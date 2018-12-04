---
type: posts
title:  "US drought maps"
classes: wide
tags: [maps, data visualization, python]
toc: false
---

This weekend exercise was born out of browsing [Weather.gov](https://www.weather.gov/) and finding out that they not only share great maps, but also make available shapefiles so that anyone can work on their results. In this post, I will look at the [United States Drought Monitor](https://droughtmonitor.unl.edu/), which gives an overview of drought conditions across the United States, with weekly updates. The website is curated by Richard Heim, at NOAA/NCEI. As always, you can find the Jupyter notebook used for the analysis on [GitHub](https://github.com/ilmonteux/mapping/).

The final results of this exercise are the US map shown below, as well as the animation , which shows the evolution of drought conditions in the continental US from 2000 to the present day. The lower panel shows what  fraction of the United States was in drought as a function of time.

<center>
<video controls>
  <source src="/assets/images/drought/US_drought_animation.mp4" type="video/mp4">
</video>
</center>

Keep reading to see how this was done, or get  the Jupyter notebook and do it yourself!

# Load data

First, let us look at what is in the shapefiles after loading them `geopandas`. We here load the current condition file, which is from last week, November 27, 2018.
```python
df = gpd.read_file('input/USDM_current_M/USDM_20181127.shp')
df.head()
```
![shapefile contents](/assets/images/drought/load_file.png)

The interesting columns are:
- `DM`, labelling the drought conditions from 0 to 4, where the former corresponds to "Abnormally dry" and the latter to "Exceptional drought".
- `geometry`, for which entries are `MultiPolygon`'s defining the geospatial attributes.
- `Shape_Leng` and `Shape_Area` giving other geometric information of the drought region such as area and length (of its boundary).

Plotting this Dataframe is easy:
```python
# yellow/orange/red colormap
cols = [plt.get_cmap('hot_r')(0.2+i/7) for i in range(5)]
# legend labels
labels = ['Normal','D0 - Abnormally dry', 'D1 - Moderate drought', 'D2 - Severe drought', 'D3 - Extreme drought', 'D4 - Exceptional drought']
# legend patches
pp = [mpl.patches.Patch(edgecolor='k', facecolor = cols[y-1] if y else 'w', label=labels[y]) for y in range(6)]

fig = plt.figure(figsize=(10,6))
ax = plt.Axes(fig, [0., 0., 1., 1.]) 
fig.add_axes(ax)
ax.set_axis_off()

# features: state borders, rivers, lakes
cdf.plot(ax=ax, edgecolor='k', facecolor=(1, 1, 1, 0), zorder=5)
riv.plot(ax=ax, color='skyblue', lw=1) #rivers
lak.plot(ax=ax, facecolor='skyblue', lw=1)
# drought regions
for i in range(5):
    df.iloc[[i]].plot(ax=ax, color=cols[i])
ax.legend(handles=pp, loc = 'lower left', frameon=False, fontsize=11)
```
![latest US drought condition map](/assets/images/drought/US_drought_20181127.png)


In order to make the map recognizable, I have also loaded other features, such as the US state borders, and datasets for rivers and lakes. The former can be found at the [US census Bureau](https://www.census.gov/geo/maps-data/data/cbf/cbf_state.html), while the latter comes from [NaturalEarthData.com](http://www.naturalearthdata.com/downloads/50m-physical-vectors/) (in particular, the "Rivers, Lake Centerlines" and the "Lakes + Reservoirs" datasets). To keep this simple, I will only visualize the continental US, forgetting about Hawaii, Alaska and Puerto Rico and other territories.

The Natural Earth datasets include features for the whole planet, which I will filter by using the `.cx` method of Geopandas, via `.cx[cdf.total_bounds[0]:cdf.total_bounds[2], cdf.total_bounds[1]:cdf.total_bounds[3]]` where `cdf` is the GeoDataFrame for the continental US. 

> There was a bug in the river dataset, where some segments of the Colorado river had coordinates such that it was in Antarctica...



## Historic data

I am interested in seeing the evolution of drought conditions as a function of time. For this, we first need to download all the data from [https://droughtmonitor.unl.edu/data/shapefiles_m/](https://droughtmonitor.unl.edu/data/shapefiles_m/). This is automated using the [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) package to parse the page, and pick up all the links to `USDM*.zip` files.

```python
url_base = 'https://droughtmonitor.unl.edu/data/shapefiles_m/'
page = requests.get(url_base).text

soup = BeautifulSoup(page, 'html.parser')
all_files = [ linkarg for node in soup.find_all('a') for linkarg in [node.contents[0]] if 
             (linkarg.endswith('.zip') and linkarg.startswith(('USDM','usdm')) and 'current' not in node.contents[0])]
all_links = [ url_base+f for f in all_files ]
```

We should now download all these files. At present, this is 987 files, each of which is about 1MB, which brings the total to about 1 GB (2 GB unzipped). Downloading all of those will take some time, so one wants to make sure we don't have to do it again. 

Therefore, I defined a function that checks the `input` directory to see if the `.zip` files and the corresponding unzipped folders are already there, in which case one can just find the shapefile in each subfolder. Running the notebook the first time, it will download all the data, unzip it and return the paths to all the available shapefiles. On a second run, it will just do the last step. If I was to run this again in a couple of weeks, it would only download the new shapefiles that were uploaded since.

How should we organize this 2 GB of data? There are multiple options, but I found the best way to define a GeoDataFrame for each drought category (five in total), with rows corresponding to the weekly observations. This way, there will be a dataframe for all `D0` observations, and so on up to a dataframe with all `D4` observations. Because this is a time series, we also add a column of dates on each dataframe.

> Unfortunately, the shapefiles are not consistently clean as I showed above. For a sizable fraction of them, there are multiple entries for each drought categories. For example, instead of having only one row for `D1`, there might be many: one needs to merge different rows, which is achieved via `.dissolve(by='DM',aggfunc='sum')`.
> GeoPandas `.dissolve()` method is the same as applying Pandas `.groupby()` and `.agg()` methods, but it also knows what to do with the geometry column of each row (in this case, union). **Note**: it is somewhat slow (one to few seconds for each shapefile in this dataset), so we only call it when necessary.

# Plotting and animating

After loading all the shapefiles into memory, we have a vector of 5 GeoDataFrames, `all_dfs`, each of which has geospatial data from 2000 to the present for each drought category, `D0-4`. A map like the one above can be made for any week since 2000. I define a function `drought_map(i, ax)` with the same commands as above, except for picking the right element of the DataFrames:

```python
    for j in range(5): # loop over drought categories
        all_dfs[j].loc[[i]].plot(ax=ax,color=cols[j])
```
Here `all_dfs[j].loc[[i]]` picks the j-th drought category and the i-th row.

Another question we can ask is: *at any given time, how much of the US was in a state of drought?*    
We can answer this question by taking the total land area in each drought condition, divide by the US land surface, and see how the result varies with time. The following figure shows a stacked area graph, with two different ways of stacking the data.

```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,4))
d = areas['date'].dt.to_pydatetime()
ax1.stackplot(d, areas.iloc[:,1:].T, colors=cols)
ax2.stackplot(d, areas.iloc[:,:0:-1].T, colors=cols[::-1]);
for ax in (ax1, ax2):
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    ax.xaxis.set_major_locator(mpl.dates.YearLocator())
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonth=range(1,13,3)))
    plt.setp(ax.get_xticklabels(),rotation=45,ha='center')
    ax.set(xlim=(d[0],d[-1]), ylim=(0,1), facecolor='0.9');
ax1.legend(handles=pp[:0:-1], loc = 'upper left', frameon=False, fontsize=10)
ax2.legend(handles=pp[1:], loc = 'upper left', frameon=False, fontsize=10)
```
![latest US drought condition map](/assets/images/drought/US_drought_timeseries.png)

> Aside: stacked graphs can be misleading, depending on the characteristcs of the underlying data. People usually plot stacked graphs with the largest contributions at the bottom and the smallest on top, but I have been going back and forth between the left and right plot and cannot decide which one I prefer. For example, the plot on the right clearly points to 2011, with about 10% of the US in state of exceptional drought: for the record, that was almost all of Texas, as well as large swaths of Oklahoma, Kansas, Louisana and New Mexico. Depending on what one cares about, it might be more useful to show the more extreme conditions at the bottom.

> Aside 2: here I also included data from Alaska, both getting amounts of land in drought conditions and for the total US area. Given that Alaska counts for 17.5% of the US land mass, and is not too often in a drought, this might skew the results.

Finally, we can make an animation that shows the evolution of drought conditions across the (continental) US (see top of this page).
For this, I used matplotlib's `animation` package, namely `FuncAnimation`. I define a `animate()` function which draws the US map by calling the previously defined `drought_map()` function, and then with `drought_series()` below it draws the time series up to that given time, similarly to how I plotted the timeseries above. The animation is then saved with `ffmpeg`.

```python 
from matplotlib import animation, rc
from IPython.display import HTML
rc('animation', html='html5')
rc('animation', embed_limit=200.)

def animate(i, axes):
    drought_map(i,axes[0])
    drought_series(i,axes[1])
    axes[1].set(yticks=[0,0.5,1], yticklabels=[])
    axes[1].tick_params('y', length=0)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_axes([0., 0.2, 1., 0.8])
ax2 = fig.add_axes([0.035, 0.07, 0.93, 0.15])

frames=range(len(all_shapefiles))
anim = animation.FuncAnimation(fig, animate, frames=frames, fargs=([ax1,ax2],), interval=12);
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
anim.save('US_drought_animation.mp4', writer = animation.FFMpegWriter(fps=26))
```

Thanks for reading! You can find the Jupyter notebook used for the analysis on [GitHub](https://github.com/ilmonteux/mapping/).

