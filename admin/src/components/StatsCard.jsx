function StatsCard({ title, value, icon, color = 'blue' }) {
  const styles = {
    blue: { bg: 'bg-blue-50', text: 'text-blue-600', accent: 'bg-blue-500' },
    green: { bg: 'bg-emerald-50', text: 'text-emerald-600', accent: 'bg-emerald-500' },
    amber: { bg: 'bg-amber-50', text: 'text-amber-600', accent: 'bg-amber-500' },
    red: { bg: 'bg-red-50', text: 'text-red-500', accent: 'bg-red-500' },
  };

  const s = styles[color];

  return (
    <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-5 hover:shadow-md transition-shadow relative overflow-hidden">
      <div className={`absolute top-0 left-0 w-full h-1 ${s.accent}`} />
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">{title}</p>
          <p className="text-3xl font-semibold text-slate-900 tabular-nums">{value}</p>
        </div>
        <div className={`p-2.5 rounded-lg ${s.bg} ${s.text}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

export default StatsCard;
