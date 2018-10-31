---
layout: splash
header:
  overlay_image: /assets/images/simon-matzinger-320332-unsplash.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [Simon Matzinger on **Unsplash**](https://unsplash.com/photos/twukN12EN7c)"

feature_row:
  - image_path: /assets/images/teasers/chronomaps.png
    alt: "NYC chronomap"
    title: "Chronomaps"
    excerpt: "Create maps based on travel times instead of distances."
    url: "chronomaps/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: /assets/images/teasers/election_county_map_carto.png
    alt: "2016 election map and cartogram"
    title: "Cartograms of US elections"
    excerpt: "Visualize county-level US election results, make *cartograms* based on county population."
    url: "cartograms/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: assets/images/teasers/LHCra.png
    alt: "Rectangular aggregations at the LHC"
    title: "Rectangular aggregations at the LHC"
    excerpt: "Data mining algorithm for statistically significant excesses in high-dimensional LHC datasets."
    url: "LHC_rectangular_aggregation"
    btn_label: "Read More"
    btn_class: "btn--primary"

---


<h1> Recent Posts </h1>
<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      - <small><strong>{{ post.date | date: "%B %e, %Y" }}</strong></small>.
    </li>
  {% endfor %}
</ul>

See all posts [here](/archive/).


<h1> Projects </h1>

{% include feature_row id="feature_row" %}


{% comment %}
This is a comment: it will not show up on the website :)
{% endcomment %}
