from dashboard.shell import render
from dashboard import live

render(live.collected_data, needs_data=True)
