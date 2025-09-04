{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    {% block methods %}
    {%- if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :toctree: .
    {% for item in all_methods %}
    {%- if item not in ['__init__'] and item not in inherited_members%}
        ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endfor %}
    {%- endif %}
    {%- endblock %}
    {% block attributes %}
    {%- if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :toctree: .
    {% for item in attributes %}
    {%- if item not in inherited_members%}
        ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endfor %}
    {%- endif %}
    {% endblock %}
