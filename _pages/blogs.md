---
title: "Blog"
layout: gridlay
sitemap: false
permalink: /blogs/
---

## ✍️ Blog

<p class="blog-intro">A home for paper reading, course notes, engineering notes, and occasional personal writing.</p>

<div class="blog-filter-bar" role="group" aria-label="Filter blog posts by category" markdown="0">
  <button class="blog-filter is-active" type="button" data-blog-filter="all" aria-pressed="true">All</button>
  {% for category in site.data.blog_categories %}
    <button class="blog-filter" type="button" data-blog-filter="{{ category.slug }}" aria-pressed="false">{{ category.title }}</button>
  {% endfor %}
</div>

{% if site.posts.size > 0 %}
  <div class="blog-post-list" aria-live="polite" markdown="0">
    {% for post in site.posts %}
      {% assign primary_category = post.categories | first | default: "personal" %}
      {% assign category_info = site.data.blog_categories | where: "slug", primary_category | first %}
      {% assign updated_at = post.last_updated | default: post.date %}
      <article class="blog-post" data-blog-categories="{{ post.categories | join: ' ' | downcase }}">
        <div class="blog-post__meta">
          <time datetime="{{ post.date | date_to_xmlschema }}">Published {{ post.date | date: "%b %-d, %Y" }}</time>
          <span aria-hidden="true">·</span>
          <time datetime="{{ updated_at | date_to_xmlschema }}">Updated {{ updated_at | date: "%b %-d, %Y" }}</time>
          <span aria-hidden="true">·</span>
          <span class="blog-post__category">{% if category_info %}{{ category_info.icon }} {{ category_info.title }}{% else %}{{ primary_category | replace: '-', ' ' }}{% endif %}</span>
          <span aria-hidden="true">·</span>
          <span>{% include blog-reading-time.html post=post %} min read</span>
        </div>
        <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        {% if post.description and post.description != "" %}<p class="blog-post__excerpt">{{ post.description | strip_html | strip_newlines | truncate: 210 }}</p>{% endif %}
        {% if post.tags and post.tags.size > 0 %}
          <div class="blog-post__tags" aria-label="Post tags">
            {% for tag in post.tags limit:4 %}{% assign tag_slug = tag | slugify %}<a class="blog-post__tag" href="{{ '/tags/#' | append: tag_slug | relative_url }}">#{{ tag }}</a>{% endfor %}
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
