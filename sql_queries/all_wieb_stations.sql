-- This query filters the GSOD station list for stations contained in WIEB's
-- member states and provinces.
WITH
wieb_member_stations AS (
    SELECT *
    FROM `bigquery-public-data.noaa_gsod.stations`
    -- WIEB members: US states
    WHERE country = 'US'
        AND state in ('WA', 'OR', 'CA', 'ID', 'NV', 'AZ', 'MT', 'WY', 'UT', 'CO', 'NM')
    
    UNION ALL 
    
    SELECT *
    FROM `bigquery-public-data.noaa_gsod.stations`
    -- WIEB members: CA provinces
    -- Canadian stations have no province labels, so I have to use lat, lon bounds.
    -- Thankfully the interior borders of Alberta and BC are very simple!
    WHERE country = 'CA'
        AND lat < 60.0 -- Alberta's and BC's northern border
        AND lon < -110.0 -- Alberta's eastern border
)
select * from wieb_member_stations
;
