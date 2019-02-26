# show spreads in the 
# ().plot.density()
# plt.show()
stats['dt_spread'] = X.groupby('id').dt.max() - X.groupby('id').dt.min()
stats['mass_spread'] = D.groupby('id').mass.max() - D.groupby('id').mass.min()
stats['rt_spread'] = D.groupby('id').rt.max() - D.groupby('id').rt.min()

plt.scatter(stats.runs_no, stats.spreads)
plt.show()

stats[np.abs(stats.spreads) < 1].groupby('runs_no').spreads.plot.density(subplots=True)
plt.show()


plot_distances_to_reference(X, s=1, grouped=True)
plt.scatter(D.dt, D.rt, c=D.charge, s=1)
plt.show()
plt.scatter(D.mass, D.rt, c=D.charge, s=1)
plt.show()
plt.scatter(D.mass, D.dt, c=D.charge, s=1)
plt.show()

# this will take a while longer..
(ggplot(D, aes(x='mass', y='dt', color='charge')) + 
    geom_point(size=1))

(ggplot(stats, aes(x='runs_no', group='runs_no', y='spreads')) + 
    geom_boxplot())

(ggplot(stats, aes(x='runs_no', group='runs_no', y='spreads')) + 
    geom_violin())




# plot the spreads as a function of median mass and median dt
stats['me_dt'] = D.groupby('id').dt.median()
stats['me_rt'] = D.groupby('id').rt.median()
stats['me_mass'] = D.groupby('id').mass.median()
stats['mass_dt_vol'] = stats.mass_spread * stats.dt_spread
# stats['binned_spread'] = pd.cut(stats.spreads, bins=3)
# stats['log_binned_spread'] = pd.cut(np.log(stats.spreads), bins=3)

(ggplot(stats, aes(x='me_mass', y='me_dt', color='np.log(mass_dt_vol)')) + 
    geom_point() +
    facet_wrap('runs_no'))

(ggplot(stats, aes(x='me_mass', y='dt_spread')) + 
    geom_density_2d() +
    facet_wrap('runs_no'))

(ggplot(stats, aes(x='me_dt', y='dt_spread')) + 
    geom_density_2d() +
    facet_wrap('runs_no'))

# So, there seem to be little dependencies between thus defined spread and some general features.
# It might be better to simply model each groups range as a functiion of the collected points.
(ggplot(stats, aes(x='me_mass', y='me_rt', color='binned_spread')) + 
    geom_density_2d() + theme_xkcd() + facet_wrap("runs_no"))

# Precisely, at least no clear cut dependence here.
(ggplot(stats, aes(x='me_mass', y='me_dt', color='binned_spread')) + 
    geom_point() + theme_xkcd() + facet_wrap("runs_no"))
(ggplot(stats, aes(x='me_mass', y='me_dt', color='log_binned_spread')) + 
    geom_point() + theme_xkcd() + facet_wrap("runs_no"))

plt.scatter(P.mass, P.dt, c=P.charge, s=10)
plt.show()
plt.scatter(unlabelled_all.mass, unlabelled_all.dt, c=unlabelled_all.charge, s=1)
plt.show()
