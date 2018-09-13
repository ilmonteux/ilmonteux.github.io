---
layout: splash
header:
  overlay_image: /assets/images/simon-matzinger-320332-unsplash.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo credit: [Simon Matzinger on **Unsplash**](https://unsplash.com/photos/twukN12EN7c)"

feature_row:
  - image_path: /assets/images/chronomaps/chronomap_NY_combined.png
    alt: "NYC chronomap"
    title: "Chronomaps"
    excerpt: "Create maps based on travel times instead of distances."
    url: "chronomaps/"
    btn_label: "Read More"
    btn_class: "btn--primary"
  - image_path: assets/images/teasers/election_county_map_carto.png
    alt: "2016 election map and cartogram"
    title: "Cartograms of US elections"
    excerpt: "Visualize county-level US election results, make *cartograms* based on county population."
    url: "cartograms/"
    btn_label: "Read More"
    btn_class: "btn--primary"

---

{% comment %}
<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      - <small><strong>{{ post.date | date: "%B %e, %Y" }}</strong></small>.
    </li>
  {% endfor %}
</ul>
{% endcomment %}

- [Chronomaps](chronomaps)
- [Cartogramming US election results](cartograms)


{% include feature_row id="feature_row" %}
