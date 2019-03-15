---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

Public articles I have made significant contributions towards are listed below. I am a coauthor on all of the CMS Collaboration's publications since fulfilling their authorship requirements on October 13, 2016, currently over 220 publications. For the full list, please see: http://inspirehep.net/author/profile/A.Buccilli.1.

<b>CMS Collaboration,</b> "Search for physics beyond the standard model in high-mass diphoton events from proton-proton collisions at sqrt(s) = 13 TeV", <i>Phys. Rev. D</i> <b>98</b> (2018), no. 9, 092001, [doi:10.1103/PhysRevD.98.092001](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.092001), [arXiv:1809.00327](https://arxiv.org/abs/1809.00327)

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
