from dashboard.shell import render
from dashboard import live

render(live.data_sources, needs_data=True)
