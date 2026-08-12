---
permalink: /
title: ""
author_profile: true
classes: wide home-page
redirect_from:
  - /about/
  - /about.html
---

{% include base_path %}

<section class="home-hero" aria-labelledby="home-title">
  <div class="home-hero__content">
    <p class="eyebrow">AI Research · [University / Lab]</p>
    <h1 id="home-title">Hi, I&rsquo;m [Name].</h1>
    <p class="home-hero__statement">I explore how multimodal and generative AI systems can reason, create, and collaborate more reliably in the real world.</p>
    <div class="home-actions">
      <a class="button-link button-link--primary" href="{{ base_path }}/cv/">View CV</a>
      <a class="button-link" href="https://github.com/" rel="noopener">GitHub</a>
    </div>
  </div>
</section>

<section class="home-section" aria-labelledby="about-heading">
  <div class="home-section__heading"><h2 id="about-heading">About</h2><span class="section-kicker">01 / INTRO</span></div>
  <p>I am an AI researcher interested in capable, efficient, and human-centered learning systems. This space is intentionally a placeholder for a concise introduction, current role, and research perspective.</p>
</section>

<section class="home-section" id="research" aria-labelledby="research-heading">
  <div class="home-section__heading"><h2 id="research-heading">Research Interests</h2><span class="section-kicker">02 / FOCUS</span></div>
  <div class="interest-tags">
    <span class="interest-tag">Multimodal LLMs</span><span class="interest-tag">Diffusion LLMs</span><span class="interest-tag">Multimodal Generation</span><span class="interest-tag">LLM Agents</span><span class="interest-tag">Efficient Training &amp; Inference</span>
  </div>
</section>

<section class="home-section" aria-labelledby="selected-heading">
  <div class="home-section__heading"><h2 id="selected-heading">Selected Research</h2><a class="text-link" href="{{ base_path }}/publications/">All publications →</a></div>
  <div class="research-list">
    <article class="research-item">
      <div class="research-item__teaser">RESEARCH / 01</div>
      <div><div class="research-item__meta">Venue / Status · Year</div><h3>Research title placeholder</h3><p>A short, precise description of the research question, method, and contribution will live here.</p><div class="item-links"><a href="#">Paper</a> · <a href="#">Code</a> · <a href="#">Project</a></div></div>
    </article>
    <article class="research-item">
      <div class="research-item__teaser">RESEARCH / 02</div>
      <div><div class="research-item__meta">Venue / Status · Year</div><h3>Another research direction</h3><p>Use this row for work in progress, a system, or a project with a lightweight teaser image later.</p><div class="item-links"><a href="#">Paper</a> · <a href="#">Code</a></div></div>
    </article>
  </div>
</section>

<section class="home-section" aria-labelledby="posts-heading">
  <div class="home-section__heading"><h2 id="posts-heading">Latest Posts</h2><a class="text-link" href="{{ base_path }}/blog/">View all posts →</a></div>
  <div class="latest-posts">
    {% for post in site.posts limit:3 %}
      {% assign category = post.categories | first | default: "research" %}
      <article class="home-post">
        <div class="post-meta"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time> · <span class="post-category">{{ category }}</span> · {% include read-time.html %}</div>
        <h3><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a></h3>
        {% if post.excerpt %}<p>{{ post.excerpt | markdownify | strip_html | truncate: 180 }}</p>{% endif %}
      </article>
    {% endfor %}
  </div>
</section>

<section class="home-section two-column-note" aria-label="News and beyond research">
  <div><div class="home-section__heading"><h2>News</h2><span class="section-kicker">03 / NOW</span></div><p>Short updates, talks, releases, and milestones can be placed here without turning the page into a feed.</p></div>
  <div><div class="home-section__heading"><h2>Beyond Research</h2><span class="section-kicker">04 / OFFLINE</span></div><p>A small place for the interests and side projects that add a human dimension to the work.</p></div>
</section>
