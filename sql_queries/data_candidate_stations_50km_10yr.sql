-- For BigQuery. Output data size is 179 MB as .csv
with
candidates as (
  select usaf, wban from UNNEST(
-- These are the usaf and wban IDs for the 235 stations within
-- 50km of priority cities and with at least 10 years of data.
-- In the data tables, the "usaf" ID is renamed "stn" for some reason.
    [
      STRUCT('691484' as usaf,'99999' as wban),
      ('722890','93136'),
      ('722901','99999'),
      ('722904','23196'),
      ('722905','99999'),
      ('722930','93107'),
      ('722955','03122'),
      ('724670','03017'),
      ('724690','23062'),
      ('724694','00450'),
      ('724935','93228'),
      ('724935','99999'),
      ('724936','23254'),
      ('724947','99999'),
      ('727870','99999'),
      ('745039','99999'),
      ('999999','04133'),
      ('999999','23012'),
      ('999999','23118'),
      ('999999','23152'),
      ('999999','23196'),
      ('999999','24244'),
      ('999999','93032'),
      ('999999','93105'),
      ('999999','93107'),
      ('742075','99999'),
      ('724956','23211'),
      ('722884','99999'),
      ('722914','99999'),
      ('994014','99999'),
      ('722974','99999'),
      ('997734','99999'),
      ('722959','99999'),
      ('745060','23239'),
      ('722919','99999'),
      ('745097','99999'),
      ('722883','99999'),
      ('722913','99999'),
      ('722958','99999'),
      ('724943','99999'),
      ('722975','93106'),
      ('999999','93106'),
      ('724950','23250'),
      ('724958','99999'),
      ('999999','23160'),
      ('999999','23169'),
      ('999999','23174'),
      ('999999','23183'),
      ('999999','23185'),
      ('999999','23188'),
      ('999999','23230'),
      ('999999','23234'),
      ('999999','24033'),
      ('999999','24131'),
      ('999999','24221'),
      ('999999','93115'),
      ('999999','23239'),
      ('722783','99999'),
      ('722784','99999'),
      ('722788','99999'),
      ('722789','99999'),
      ('722885','99999'),
      ('722887','99999'),
      ('722903','99999'),
      ('722907','99999'),
      ('722927','99999'),
      ('722956','99999'),
      ('722976','99999'),
      ('724666','99999'),
      ('724699','99999'),
      ('725846','99999'),
      ('726813','99999'),
      ('726836','99999'),
      ('726959','99999'),
      ('726986','99999'),
      ('727834','99999'),
      ('727856','99999'),
      ('727928','99999'),
      ('727937','99999'),
      ('727938','99999'),
      ('747043','99999'),
      ('724937','99999'),
      ('724938','99999'),
      ('727934','99999'),
      ('726985','99999'),
      ('999999','04136'),
      ('999999','04236'),
      ('999999','53131'),
      ('999999','94074'),
      ('720565','99999'),
      ('720619','99999'),
      ('749167','99999'),
      ('994036','99999'),
      ('997292','99999'),
      ('999999','23272'),
      ('999999','94290'),
      ('994016','99999'),
      ('994018','99999'),
      ('994027','99999'),
      ('994028','99999'),
      ('994033','99999'),
      ('994034','99999'),
      ('994035','99999'),
      ('998013','99999'),
      ('994048','99999'),
      ('994041','99999'),
      ('998197','99999'),
      ('998474','99999'),
      ('998475','99999'),
      ('998476','99999'),
      ('998477','99999'),
      ('998479','99999'),
      ('998256','99999'),
      ('720339','00121'),
      ('720549','00171'),
      ('720646','00228'),
      ('720734','00264'),
      ('720741','00269'),
      ('720839','00279'),
      ('722096','53127'),
      ('722740','23160'),
      ('722745','23109'),
      ('722749','53128'),
      ('722780','23183'),
      ('722783','03185'),
      ('722784','03184'),
      ('722785','23111'),
      ('722786','23104'),
      ('722787','53126'),
      ('722788','03186'),
      ('722789','03192'),
      ('722874','93134'),
      ('722880','23152'),
      ('722885','93197'),
      ('722886','23130'),
      ('722887','03180'),
      ('722900','23188'),
      ('722903','03131'),
      ('722904','03178'),
      ('722906','93112'),
      ('722907','53143'),
      ('722909','93115'),
      ('722927','03177'),
      ('722931','93107'),
      ('722950','23174'),
      ('722955','03174'),
      ('722956','03167'),
      ('722970','23129'),
      ('722975','53141'),
      ('722976','03166'),
      ('723647','03034'),
      ('723650','23050'),
      ('723860','23169'),
      ('724666','93067'),
      ('724695','23036'),
      ('724846','53123'),
      ('724880','23185'),
      ('724930','23230'),
      ('724937','23289'),
      ('724938','93231'),
      ('724940','23234'),
      ('724950','23254'),
      ('724955','93227'),
      ('725640','24018'),
      ('725650','03017'),
      ('725720','24127'),
      ('725750','24126'),
      ('725755','24101'),
      ('725846','93201'),
      ('725850','93228'),
      ('726770','24033'),
      ('726810','24131'),
      ('726813','94195'),
      ('726836','04201'),
      ('726930','24221'),
      ('726959','94281'),
      ('726980','24229'),
      ('726985','24242'),
      ('726986','94261'),
      ('727834','24136'),
      ('727850','24157'),
      ('727855','24114'),
      ('727856','94176'),
      ('727870','94119'),
      ('727918','94298'),
      ('727928','94263'),
      ('727930','24233'),
      ('727934','94248'),
      ('727935','24234'),
      ('727937','24222'),
      ('727938','94274'),
      ('745056','53120'),
      ('745057','53130'),
      ('746140','23112'),
      ('747043','03165'),
      ('720533','00160'),
      ('720538','00164'),
      ('724699','03065'),
      ('711120','99999'),
      ('711190','99999'),
      ('711240','99999'),
      ('711260','99999'),
      ('717850','99999'),
      ('718776','99999'),
      ('718790','99999'),
      ('996870','99999'),
      ('710370','99999'),
      ('710670','99999'),
      ('711210','99999'),
      ('711230','99999'),
      ('711233','99999'),
      ('711270','99999'),
      ('711550','99999'),
      ('711570','99999'),
      ('712350','99999'),
      ('712380','99999'),
      ('712840','99999'),
      ('713510','99999'),
      ('713930','99999'),
      ('714950','99999'),
      ('715040','99999'),
      ('716050','99999'),
      ('716080','99999'),
      ('716380','99999'),
      ('717840','99999'),
      ('718600','99999'),
      ('718770','99999'),
      ('718920','99999'),
      ('719820','99999'),
      ('710420','99999'),
      ('712010','99999'),
      ('712110','99999'),
      ('717720','99999'),
      ('717750','99999'),
      ('712090','99999')
    ]
  )
)
select
  g.stn,
  g.wban,
  year,
  mo,
  da,
  temp,
  count_temp,
  dewp,
  count_dewp,
  slp,
  count_slp,
  stp,
  count_stp,
  visib,
  count_visib,
  wdsp,
  count_wdsp,
  --mxpsd,
  --gust,
  max,
  flag_max,
  min,
  flag_min,
  prcp,
  flag_prcp,
  sndp,
  --fog,
  rain_drizzle,
  snow_ice_pellets,
  hail,
  --thunder,
  --tornado_funnel_cloud
FROM `bigquery-public-data.noaa_gsod.gsod*` as g
inner join candidates as c
on g.stn = c.usaf
and g.wban = c.wban
order by 1, 2, 3, 4, 5
;
