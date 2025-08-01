-- Script per lo schema completo e aggiornato del database ForVARD

BEGIN;

-- ----------------------------
-- Tabella: roles
-- Descrizione: Definisce i ruoli degli utenti (es. base, senator, admin).
-- ----------------------------
CREATE TABLE IF NOT EXISTS public.roles
(
    role_id serial NOT NULL,
    role_name character varying(50) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT roles_pkey PRIMARY KEY (role_id),
    CONSTRAINT roles_role_name_key UNIQUE (role_name)
);

COMMENT ON TABLE public.roles
    IS 'Definisce i ruoli degli utenti (es. base, senator, admin).';

-- ----------------------------
-- Tabella: users
-- Descrizione: Memorizza le informazioni degli utenti, inclusa la mail e la password hashata.
-- ----------------------------
CREATE TABLE IF NOT EXISTS public.users
(
    user_id uuid NOT NULL DEFAULT gen_random_uuid(),
    email character varying(255) COLLATE pg_catalog."default" NOT NULL,
    password_hash character varying(255) COLLATE pg_catalog."default" NOT NULL,
    role_id integer NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT users_pkey PRIMARY KEY (user_id),
    CONSTRAINT users_email_key UNIQUE (email)
);

COMMENT ON TABLE public.users
    IS 'Memorizza le informazioni degli utenti, inclusa la mail e la password hashata.';

-- ----------------------------
-- Tabella: assets
-- Descrizione: Anagrafica di tutti gli asset finanziari disponibili.
-- ----------------------------
CREATE TABLE IF NOT EXISTS public.assets
(
    asset_id serial NOT NULL,
    symbol character varying(20) COLLATE pg_catalog."default" NOT NULL,
    description text COLLATE pg_catalog."default",
    industry character varying(255) COLLATE pg_catalog."default",
    sector character varying(255) COLLATE pg_catalog."default",
    asset_category character varying(50) COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT assets_pkey PRIMARY KEY (asset_id),
    CONSTRAINT assets_symbol_key UNIQUE (symbol)
);

COMMENT ON TABLE public.assets
    IS 'Elenco di tutti gli asset finanziari disponibili.';

-- ----------------------------
-- Tabella: asset_permissions
-- Descrizione: Mappa quali utenti hanno accesso a quali specifici asset.
-- ----------------------------
CREATE TABLE IF NOT EXISTS public.asset_permissions
(
    permission_id serial NOT NULL,
    user_id uuid NOT NULL,
    asset_id integer NOT NULL,
    CONSTRAINT asset_permissions_pkey PRIMARY KEY (permission_id),
    CONSTRAINT asset_permissions_user_id_asset_id_key UNIQUE (user_id, asset_id)
);

COMMENT ON TABLE public.asset_permissions
    IS 'Mappa quali utenti hanno accesso a quali asset.';

-- ----------------------------
-- Tabella: realized_volatility_data
-- Descrizione: Contiene i dati di volatilità giornalieri calcolati.
-- ----------------------------
CREATE TABLE IF NOT EXISTS public.realized_volatility_data
(
    id serial NOT NULL,
    observation_date date NOT NULL,
    symbol character varying(20) COLLATE pg_catalog."default" NOT NULL,
    asset_type character varying(50) COLLATE pg_catalog."default",
    volume bigint,
    trades integer,
    open_price numeric,
    close_price numeric,
    high_price numeric,
    low_price numeric,
    pv numeric,
    gk numeric,
    rr5 numeric,
    rv1 numeric,
    rv5 numeric,
    rv5_ss numeric,
    bv1 numeric,
    bv5 numeric,
    bv5_ss numeric,
    rsp1 numeric,
    rsn1 numeric,
    rsp5 numeric,
    rsn5 numeric,
    rsp5_ss numeric,
    rsn5_ss numeric,
    medrv1 numeric,
    medrv5 numeric,
    medrv5_ss numeric,
    minrv1 numeric,
    minrv5 numeric,
    minrv5_ss numeric,
    rk numeric,
    rq1 numeric, -- Nuova colonna
    rq5 numeric, -- Nuova colonna
    rq5_ss numeric, -- Nuova colonna
    CONSTRAINT realized_volatility_data_pkey PRIMARY KEY (id),
    CONSTRAINT realized_volatility_data_observation_date_symbol_key UNIQUE (observation_date, symbol)
);

COMMENT ON TABLE public.realized_volatility_data
    IS 'Contiene i dati di volatilità giornalieri calcolati.';
COMMENT ON COLUMN public.realized_volatility_data.rq1
    IS 'Realized Quarticity (1-min)';
COMMENT ON COLUMN public.realized_volatility_data.rq5
    IS 'Realized Quarticity (5-min)';
COMMENT ON COLUMN public.realized_volatility_data.rq5_ss
    IS 'Realized Quarticity (5-min, sub-sampled)';

-- ----------------------------
-- Tabella: realized_covariance_data
-- Descrizione: Contiene i dati di covarianza giornalieri calcolati.
-- ----------------------------
CREATE TABLE IF NOT EXISTS public.realized_covariance_data
(
    id serial NOT NULL,
    date date NOT NULL,
    asset1 character varying(20) COLLATE pg_catalog."default" NOT NULL,
    asset2 character varying(20) COLLATE pg_catalog."default" NOT NULL,
    rcov numeric,
    rbpcov numeric,
    rscov_p numeric,
    rscov_n numeric,
    rscov_mp numeric,
    rscov_mn numeric,
    CONSTRAINT realized_covariance_data_pkey PRIMARY KEY (id),
    CONSTRAINT realized_covariance_data_date_asset1_ass_key UNIQUE (date, asset1, asset2)
);

COMMENT ON TABLE public.realized_covariance_data
    IS 'Contiene i dati di covarianza giornalieri calcolati.';

-- ----------------------------
-- Funzioni e Trigger (invariati)
-- ----------------------------
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_timestamp_users
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION trigger_set_timestamp();

-- ----------------------------
-- Definizione Foreign Keys
-- ----------------------------
ALTER TABLE IF EXISTS public.users
    ADD CONSTRAINT users_role_id_fkey FOREIGN KEY (role_id)
    REFERENCES public.roles (role_id) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE NO ACTION;

ALTER TABLE IF EXISTS public.asset_permissions
    ADD CONSTRAINT asset_permissions_asset_id_fkey FOREIGN KEY (asset_id)
    REFERENCES public.assets (asset_id) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE CASCADE;

ALTER TABLE IF EXISTS public.asset_permissions
    ADD CONSTRAINT asset_permissions_user_id_fkey FOREIGN KEY (user_id)
    REFERENCES public.users (user_id) MATCH SIMPLE
    ON UPDATE NO ACTION
    ON DELETE CASCADE;

-- ----------------------------
-- Creazione Indici
-- ----------------------------
CREATE INDEX IF NOT EXISTS idx_rv_data_date_symbol ON public.realized_volatility_data (observation_date, symbol);
CREATE INDEX IF NOT EXISTS idx_rv_data_symbol ON public.realized_volatility_data (symbol);
CREATE INDEX IF NOT EXISTS idx_rv_data_asset_type ON public.realized_volatility_data (asset_type);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users (email);
CREATE INDEX IF NOT EXISTS idx_users_role_id ON public.users (role_id);


END;