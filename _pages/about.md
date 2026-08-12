---
layout: home
permalink: /
title: ""
author_profile: false
redirect_from:
  - /about/
  - /about.html
---

{% include base_path %}

<section class="editorial-hero" aria-labelledby="home-title">
  <div class="editorial-hero__copy">
    <p class="eyebrow">AI Research <span>·</span> Peking University</p>
    <h1 id="home-title">Hi, I&rsquo;m<br><em>Yihan Wang.</em><span class="hero-name-cn">汪轶寒</span></h1>
    <p class="editorial-hero__statement">I explore how foundation models learn, generate, reason, and interact with the world.</p>
    <p class="editorial-hero__focus">Multimodal LLMs <span>·</span> Diffusion LLMs <span>·</span> Agents <span>·</span> Efficient AI</p>
    <div class="hero-actions">
      <a class="button-link button-link--primary" href="#research">Research interests <span>↓</span></a>
      <a class="button-link" href="{{ base_path }}/cv/">CV <span>↗</span></a>
    </div>
    <div class="hero-socials" aria-label="Social links">
      <a href="https://github.com/Anonymous-yh" rel="noopener">GitHub</a><a href="mailto:pkuwyh0131@gmail.com">Email</a>
    </div>
  </div>
  <div class="editorial-hero__portrait editorial-hero__portrait--placeholder" aria-label="Profile photo will be added soon">
    <div class="portrait-grid" aria-hidden="true"></div>
    <div class="portrait-placeholder" aria-hidden="true"><span>YW</span></div>
    <span class="portrait-label">PHOTO / TO BE ADDED</span>
  </div>
</section>

<section class="interest-strip" id="research" aria-labelledby="interests-heading">
  <p id="interests-heading" class="section-number">00 / RESEARCH INTERESTS</p>
  <div class="interest-strip__items"><span>Multimodal LLMs</span><span>Diffusion LLMs</span><span>Generative Models</span><span>AI Agents</span><span>Efficient Inference</span></div>
</section>

<section class="editorial-section about-section" aria-labelledby="about-heading">
  <div class="section-heading"><p class="section-number">01 / ABOUT</p><h2 id="about-heading">Building toward capable AI.</h2></div>
  <div class="about-grid">
    <p class="about-grid__lead">I am an undergraduate student majoring in Intelligence Science and Technology at Peking University, and will join the School of Intelligence at Peking University as a Ph.D. student in Fall 2027.</p>
    <div><p>My interests lie broadly in foundation models, particularly multimodal large language models, diffusion language models, multimodal generation and understanding, intelligent agents, and efficient model training and inference.</p><a class="text-link" href="{{ base_path }}/cv/">More about my background →</a></div>
  </div>
</section>

<section class="editorial-section news-section" aria-labelledby="news-heading">
  <div class="section-heading"><p class="section-number">02 / NEWS</p><h2 id="news-heading">What&rsquo;s happening.</h2></div>
  <div class="news-list">
    <div class="news-item"><time>2027.09</time><p>Will join the School of Intelligence, Peking University as a Ph.D. student.</p></div>
    <div class="news-item"><time>2026.08</time><p>Exploring few-step generation and step distillation for diffusion language models.</p></div>
    <div class="news-item"><time>2026.07</time><p>Studying efficient inference and acceleration methods for diffusion language models.</p></div>
  </div>
</section>

<section class="editorial-section writing-section" aria-labelledby="writing-heading">
  <div class="section-heading section-heading--spread"><div><p class="section-number">03 / WRITING</p><h2 id="writing-heading">Notes in progress.</h2></div><a class="text-link" href="{{ base_path }}/blog/">View all writing →</a></div>
  <p class="section-intro">Notes on AI research, engineering, and things I find interesting.</p>
  <div class="writing-list">
    {% for post in site.posts limit:3 %}
      {% assign category = post.categories | first | default: "research" %}
      {% case category %}{% when "research" %}{% assign category_label = "Research Notes" %}{% when "modding" %}{% assign category_label = "Game Modding" %}{% else %}{% assign category_label = category | capitalize %}{% endcase %}
      <article class="writing-entry"><div class="writing-entry__meta"><span>{{ category_label }}</span><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%b %-d, %Y" }}</time></div><div><h3><a href="{{ base_path }}{{ post.url }}">{{ post.title }}</a></h3>{% if post.excerpt %}<p>{{ post.excerpt | markdownify | strip_html | truncate: 160 }}</p>{% endif %}</div><div class="writing-entry__read">{% include read-time.html %}<a href="{{ base_path }}{{ post.url }}">Read →</a></div></article>
    {% endfor %}
    {% if site.posts.size == 0 %}
      <p class="empty-state">Writing is coming soon. I plan to share research notes, engineering experiments, and game-modding explorations here.</p>
    {% endif %}
  </div>
</section>

<section class="beyond-section" aria-labelledby="beyond-heading"><p class="section-number">04 / BEYOND RESEARCH</p><h2 id="beyond-heading">Curiosity extends beyond the lab.</h2><p>Outside research, I enjoy game modding, games, and exploring how interactive systems are built.</p></section>
