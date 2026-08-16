---
title: "Blog"
layout: gridlay
sitemap: false
permalink: /blogs/
---

## ✍️ Blog

<p class="blog-intro">A home for paper reading, course notes, engineering notes, and occasional personal writing.</p>

<div class="blog-category-overview" markdown="0">
  {% for category in site.data.blog_categories %}
    {% assign post_count = site.categories[category.slug] | size %}
    <div class="blog-category-overview__item">
      <span class="blog-category-overview__icon" aria-hidden="true">{{ category.icon }}</span>
      <div>
        <strong>{{ category.title }}</strong>
        <p>{{ category.description }}</p>
      </div>
      <span class="blog-category-overview__count">{{ post_count }}</span>
    </div>
  {% endfor %}
</div>

<div class="blog-filter-bar" role="group" aria-label="Filter blog posts by category" markdown="0">
  <button class="blog-filter is-active" type="button" data-blog-filter="all" aria-pressed="true">All <span>{{ site.posts | size }}</span></button>
  {% for category in site.data.blog_categories %}
    {% assign post_count = site.categories[category.slug] | size %}
    <button class="blog-filter" type="button" data-blog-filter="{{ category.slug }}" aria-pressed="false">{{ category.icon }} {{ category.title }} <span>{{ post_count }}</span></button>
  {% endfor %}
</div>

{% if site.posts.size > 0 %}
  <div class="blog-post-list" aria-live="polite" markdown="0">
    {% for post in site.posts %}
      {% assign primary_category = post.categories | first | default: "personal" %}
      {% assign category_info = site.data.blog_categories | where: "slug", primary_category | first %}
      {% assign reading_time = post.content | number_of_words | divided_by: 220 | plus: 1 %}
      <article class="blog-post" data-blog-categories="{{ post.categories | join: ' ' | downcase }}">
        <div class="blog-post__meta">
          <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time>
          <span aria-hidden="true">·</span>
          <span class="blog-post__category">{% if category_info %}{{ category_info.icon }} {{ category_info.title }}{% else %}{{ primary_category | replace: '-', ' ' }}{% endif %}</span>
          <span aria-hidden="true">·</span>
          <span>{{ reading_time }} min read</span>
        </div>
        <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        {% if post.description and post.description != "" %}<p class="blog-post__excerpt">{{ post.description | strip_html | strip_newlines | truncate: 210 }}</p>{% endif %}
        {% if post.tags and post.tags.size > 0 %}
          <div class="blog-post__tags" aria-label="Post tags">
            {% for tag in post.tags limit:4 %}<span>#{{ tag }}</span>{% endfor %}
          </div>
        {% endif %}
      </article>
    {% endfor %}
  </div>
  <p class="blog-filter-empty" hidden>No posts in this category yet.</p>
{% else %}
  <div class="blog-empty-state" markdown="0">
    <span aria-hidden="true">✦</span>
    <p>No posts yet. The first notes are on their way.</p>
  </div>
{% endif %}
